"""
Offline Distributed Rollout Worker for player-perspective Orbit Wars imitation.

This version matches the asymmetric/player-perspective observation architecture:
  - one trajectory is submitted per present replay player (games with no
    curated players are skipped). Curated players carry real BC labels;
    non-curated players carry sentinel ILLEGAL labels (target_idx == -1)
    so the learner trains only the value head on them;
  - each trajectory uses OrbitWarsAssembler.reset(obs0, player_id) so owner
    encodings are perspective-relative;
  - rewards, values, log-probs, and advantages are scalar per player sample;
  - the packed observation layout is
        [global_features, planet_features, edge_features, edge_mask,
         planet_mask, source_mask]
    with no canonical owner_ids side-channel.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import math
import os
import queue
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray

from config import RunConfig
from orbit_obs_reftrace import OrbitWarsAssembler, schema_metadata
from orbit_obs_utils_reftrace import MAX_PLAYERS, OPT_MAX_SEND, OPT_SKIP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SYNC LEARNER CLIENT
# ---------------------------------------------------------------------------

class SyncLearnerClient:
    def __init__(self, learner_actor, cfg: RunConfig):
        self.learner_actor = learner_actor
        self.cfg = cfg.rollout
        self.q = queue.Queue(
            maxsize=self.cfg.learn_max_pending_batches * self.cfg.learn_max_episodes
        )
        threading.Thread(target=self._worker, daemon=True).start()

    def submit_episode(
        self,
        obs: np.ndarray,    # [T, obs_dim]
        act: np.ndarray,    # [T, P, 2]
        logp: np.ndarray,   # [T]
        val: np.ndarray,    # [T]
        var: np.ndarray,    # [T]
        rew: np.ndarray,    # [T]
        done: np.ndarray,   # [T]
    ) -> bool:
        if self.q.full():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put_nowait((obs, act, logp, val, var, rew, done))
        return True

    def _worker(self):
        while True:
            items = [self.q.get()]
            while len(items) < self.cfg.learn_max_episodes:
                try:
                    items.append(self.q.get_nowait())
                except queue.Empty:
                    break
            packed = self._prepare_batch(items)
            try:
                ray.get(self.learner_actor.submit_packed_batch.remote(*packed))
            except Exception as e:
                logger.error(f"Learner submission failed: {e}")

    @staticmethod
    def _prepare_batch(items):
        lengths = np.asarray([it[1].shape[0] for it in items], dtype=np.int32)
        return (
            np.concatenate([it[0] for it in items], axis=0).astype(np.float32),
            np.concatenate([it[1] for it in items], axis=0).astype(np.int64),
            np.concatenate([it[2] for it in items], axis=0).astype(np.float32),
            np.concatenate([it[3] for it in items], axis=0).astype(np.float32),
            np.concatenate([it[4] for it in items], axis=0).astype(np.float32),
            np.concatenate([it[5] for it in items], axis=0).astype(np.float32),
            np.concatenate([it[6] for it in items], axis=0).astype(np.float32),
            lengths,
        )


# ---------------------------------------------------------------------------
# UTILITIES & REVERSE-MAPPING HELPERS
# ---------------------------------------------------------------------------

@dataclass
class MatchOutcome:
    terminal_reward: float


def _production_vector(obs: Dict[str, Any], max_players: int) -> np.ndarray:
    """Production currently controlled by each real player slot."""
    out = np.zeros(int(max_players), dtype=np.float32)
    for p in obs.get("planets", []) or []:
        owner = int(p[1])
        prod = float(p[6])
        if prod > 0.0 and 0 <= owner < int(max_players):
            out[owner] += prod
    return out


def _total_production_capacity(obs: Dict[str, Any]) -> float:
    """Total positive production available on the map, including neutral planets."""
    total = 0.0
    for p in obs.get("planets", []) or []:
        prod = float(p[6])
        if prod > 0.0:
            total += prod
    return max(1.0, float(total))


def _production_advantage_raw(
    obs: Dict[str, Any],
    *,
    max_players: int,
    n_agents: int,
    total_prod: float,
) -> np.ndarray:
    """Map-normalized competitive production margin.

    raw_i = (own_i - max(enemy_i)) / total_map_production

    This is bounded in [-1, +1]. Rewards are computed as raw deltas,
    so starting-map asymmetry does not itself create reward. In FFA this
    rewards gaining ground against the strongest opponent rather than against
    the sum of all opponents.
    """
    max_players = int(max_players)
    active = max(1, min(int(n_agents), max_players))
    total_prod = max(1.0, float(total_prod))

    prod = _production_vector(obs, max_players)
    raw = np.zeros(max_players, dtype=np.float32)

    active_prod = prod[:active].astype(np.float32, copy=False)
    if active <= 1:
        enemy_max = np.zeros(active, dtype=np.float32)
    else:
        # active is at most MAX_PLAYERS (normally 2 or 4), so this explicit
        # loop is clearer and negligible compared with observation assembly.
        enemy_max = np.zeros(active, dtype=np.float32)
        for pid in range(active):
            enemy_prod = np.delete(active_prod, pid)
            enemy_max[pid] = float(np.max(enemy_prod)) if enemy_prod.size else 0.0

    raw[:active] = (active_prod - enemy_max) / total_prod

    return np.clip(raw, -1.0, 1.0).astype(np.float32)


def iter_replay_paths(data_dir: str):
    search_pattern = os.path.join(os.path.abspath(data_dir), "**", "*.json")
    yield from sorted(glob.iglob(search_pattern, recursive=True))


def replay_key(data_dir: str, file_path: str) -> str:
    root = os.path.abspath(data_dir)
    path = os.path.abspath(file_path)
    return os.path.relpath(path, root).replace(os.sep, "/")


def shard_for_key(key: str, num_workers: int) -> int:
    if num_workers <= 1:
        return 0
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="little", signed=False) % num_workers


def load_replay_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "steps" in data and len(data["steps"]) > 0:
        return data
    return None


def _norm_player_name(name: str) -> str:
    return str(name).strip().casefold()


def load_player_allowlist(path: str) -> set[str]:
    """Loads curated player names from a text file, one name per line."""
    if not path:
        return set()

    if not os.path.exists(path):
        logger.warning(
            f"Curated player allowlist not found at {path!r}; "
            "offline worker will train on all players."
        )
        return set()

    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.add(_norm_player_name(line))

    logger.info(f"Loaded {len(out)} curated player name(s) from {path!r}")
    return out


def replay_player_names(replay: dict) -> list[str]:
    """
    Returns player names by slot. Preferred source:
      replay["info"]["Agents"][slot]["Name"]

    Fallback:
      replay["info"]["TeamNames"][slot]
    """
    info = replay.get("info") or {}

    agents = info.get("Agents") or []
    names = []
    for a in agents:
        if isinstance(a, dict):
            name = a.get("Name")
            if name is not None:
                names.append(str(name))

    if names:
        return names

    team_names = info.get("TeamNames") or []
    if team_names:
        return [str(x) for x in team_names]

    return []


def curated_player_slots(replay: dict, allowed_names: set[str]) -> set[int]:
    """
    Returns player slots whose names are in the curated allowlist.
    If allowed_names is empty, all discovered slots are allowed.
    """
    names = replay_player_names(replay)

    if not allowed_names:
        return set(range(len(names))) if names else set(range(MAX_PLAYERS))

    allowed_slots = set()
    for slot, name in enumerate(names):
        if _norm_player_name(name) in allowed_names:
            allowed_slots.add(slot)

    return allowed_slots


def replay_rewards(replay: dict, last_step) -> list[float]:
    """
    Return final rewards by player slot.

    Preferred:
      replay["rewards"]

    Fallback:
      steps[-1][slot]["reward"]
    """
    top_rewards = replay.get("rewards", None)
    if isinstance(top_rewards, list) and len(top_rewards) > 0:
        return [float(r or 0.0) for r in top_rewards]

    out = []
    for entry in last_step:
        if isinstance(entry, dict):
            out.append(float(entry.get("reward", 0.0) or 0.0))
        else:
            out.append(float(getattr(entry, "reward", 0.0) or 0.0))
    return out


def pack_observation_np(d: Dict[str, Any]) -> np.ndarray:
    """NumPy-only flat packer matching ppo_core.OrbitObservationUnpacker.

    Asymmetric/player-perspective layout:
      global_features, planet_features, edge_features, edge_mask, planet_mask,
      source_mask

    ``source_mask`` is already perspective-relative because each assembler is
    reset with a concrete ``player_id``.
    """

    return np.concatenate(
        [
            np.asarray(d["global_features"], dtype=np.float32).reshape(-1),
            np.asarray(d["planet_features"], dtype=np.float32).reshape(-1),
            np.asarray(d["edge_features"], dtype=np.float32).reshape(-1),
            np.asarray(d["edge_mask"], dtype=np.float32).reshape(-1),
            np.asarray(d["planet_mask"], dtype=np.float32).reshape(-1),
            np.asarray(d["source_mask"], dtype=np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def terminal_reward_vector(
    *,
    winner: Optional[int],
    n_present: int,
    cfg: RunConfig,
    max_players: int,
) -> np.ndarray:
    """Build terminal reward vector indexed by raw player id 0..3."""
    out = np.zeros(max_players, dtype=np.float32)
    n = max(0, min(int(n_present), int(max_players)))
    if n <= 0:
        return out

    win_r = float(cfg.reward.terminal_win)
    loss_r = float(cfg.reward.terminal_loss)
    for pid in range(n):
        out[pid] = win_r if (winner is not None and pid == int(winner)) else loss_r
    return out


# ---------------------------------------------------------------------------
# OFFLINE ROLLOUT WORKER
# ---------------------------------------------------------------------------

class OfflineRolloutWorker:
    def __init__(
        self,
        cfg: RunConfig,
        inference_actor,
        learner_actor,
        data_dir: str = "./data",
        pairs: int = 1,
        server_port: int = 0,
        worker_id: int = 0,
        num_workers: int = 1,
        refresh_sec: float = 60.0,
        curated_players_path: str = "./curated_players.txt",
    ):
        self.cfg = cfg
        self.data_dir = data_dir
        self.learner_client = SyncLearnerClient(learner_actor, cfg)

        # Offline imitation does not use inference, but the argument is kept
        # for orchestrator/Ray wiring compatibility.
        self.inference_actor = inference_actor

        self._traj: Dict[str, Dict[str, list]] = {}
        self._meta = schema_metadata()
        self._max_planets = int(self._meta["max_planets"])
        self._max_players = int(self._meta.get("max_players", MAX_PLAYERS))

        self.production_coef = float(getattr(cfg.reward, "production_coef", 0.05))
        self.imitation_chunk_len = int(getattr(cfg.rollout, "imitation_chunk_len", 32))

        self.use_heuristic_expert = True
        self.worker_id = int(worker_id)
        self.num_workers = max(1, int(num_workers))
        self.refresh_sec = max(1.0, float(refresh_sec))
        self.curated_players_path = str(curated_players_path)
        self.curated_player_names = load_player_allowlist(self.curated_players_path)
        self._last_epoch_curated_slots = 0
        self._last_epoch_value_only_slots = 0
        self._last_epoch_skipped_no_curated = 0
        self._last_epoch_skipped_player_turns = 0
        self._matches_processed = 0
        self._epochs_completed = 0
        self._last_epoch_found = 0
        self._last_epoch_processed = 0

    def run(self):
        """Persistent shuffled offline replay epochs."""
        print(
            f"[*] Offline worker {self.worker_id}/{self.num_workers} initialized. "
            f"Ingesting from {self.data_dir}; sleep_between_epochs={self.refresh_sec}"
        )

        rng = np.random.default_rng(12345 + self.worker_id)

        while True:
            try:
                processed = self._run_offline_epoch_once(rng=rng)
                self._epochs_completed += 1

                print(
                    f"[*] Offline worker {self.worker_id}: "
                    f"epoch={self._epochs_completed}, "
                    f"assigned={self._last_epoch_found}, "
                    f"processed={processed}, "
                    f"curated_slots={self._last_epoch_curated_slots}, "
                    f"value_only_slots={self._last_epoch_value_only_slots}, "
                    f"skipped_no_curated={self._last_epoch_skipped_no_curated}, "
                    f"total_processed={self._matches_processed}"
                )

            except Exception as e:
                logger.exception(f"Offline worker {self.worker_id} epoch failed: {e}")

            time.sleep(self.refresh_sec)

    def _submit_imitation_sequence(self, obs_list, act_list, rew_list, done_last):
        T = len(obs_list)
        if T <= 0:
            return

        obs_stacked = np.ascontiguousarray(np.stack(obs_list, axis=0), dtype=np.float32)
        act_stacked = np.ascontiguousarray(np.stack(act_list, axis=0), dtype=np.int64)
        rewards = np.ascontiguousarray(np.stack(rew_list, axis=0), dtype=np.float32)

        dones = np.zeros(T, dtype=np.float32)
        if done_last:
            dones[-1] = 1.0

        zeros_logp = np.zeros(T, dtype=np.float32)
        zeros_val = np.zeros(T, dtype=np.float32)

        self.learner_client.submit_episode(
            obs_stacked,
            act_stacked,
            zeros_logp,
            zeros_val,
            zeros_val.copy(),
            rewards,
            dones,
        )

    def _flush_imitation_ready_chunks(self):
        chunk_len = max(2, int(self.imitation_chunk_len))

        for tag, buf in list(self._traj.items()):
            ready = min(len(buf["obs"]), len(buf["act"]), len(buf["rew"]))
            while ready >= chunk_len:
                n = chunk_len
                self._submit_imitation_sequence(
                    buf["obs"][:n], buf["act"][:n], buf["rew"][:n], done_last=False,
                )
                del buf["obs"][:n], buf["act"][:n], buf["rew"][:n]
                ready = min(len(buf["obs"]), len(buf["act"]), len(buf["rew"]))

    def _finalize_imitation_trajectory(self, tag: str, outcome: MatchOutcome):
        buf = self._traj.pop(tag, None)
        if buf is None or not buf.get("act"):
            return

        ready = min(len(buf["obs"]), len(buf["act"]), len(buf["rew"]))
        if ready <= 0:
            return

        # 1. Add the terminal win/loss reward to the final frame
        buf["rew"][ready - 1] = np.float32(
            float(buf["rew"][ready - 1]) + float(outcome.terminal_reward)
        )

        # 2. Submit the RAW rewards!
        # By submitting the full episode at once, learner.py will successfully
        # backpropagate this terminal reward all the way to step 0 using GAE.
        self._submit_imitation_sequence(
            buf["obs"][:ready], buf["act"][:ready], buf["rew"][:ready], done_last=True,
        )

    def _run_offline_epoch_once(self, rng: np.random.Generator) -> int:
        processed_this_epoch = 0
        self._last_epoch_curated_slots = 0
        self._last_epoch_value_only_slots = 0
        self._last_epoch_skipped_no_curated = 0
        self._last_epoch_skipped_player_turns = 0

        # Rescan every epoch so newly uploaded .json files are picked up.
        all_paths = list(iter_replay_paths(self.data_dir))

        assigned_paths = []
        for file_path in all_paths:
            key = replay_key(self.data_dir, file_path)
            if shard_for_key(key, self.num_workers) == self.worker_id:
                assigned_paths.append(file_path)

        rng.shuffle(assigned_paths)
        self._last_epoch_found = len(assigned_paths)

        for file_path in assigned_paths:
            try:
                replay = load_replay_file(file_path)
            except json.JSONDecodeError:
                # File may still be copying. Try again next epoch.
                continue
            except OSError:
                # File may be temporarily locked. Try again next epoch.
                continue

            if replay is None:
                continue

            steps = replay.get("steps", [])
            if len(steps) < 2:
                continue

            allowed_slots = curated_player_slots(replay, self.curated_player_names)
            allowed_slots = {int(s) for s in allowed_slots if 0 <= int(s) < self._max_players}
            if not allowed_slots:
                self._last_epoch_skipped_no_curated += 1
                continue

            self._last_epoch_curated_slots += len(allowed_slots)

            last_step = steps[-1]
            rewards = replay_rewards(replay, last_step)
            n_present = min(len(steps[0]), self._max_players)
            if rewards:
                top_reward = max(rewards)
                top_slots = [i for i, r in enumerate(rewards[:self._max_players]) if r == top_reward]
                winner = top_slots[0] if len(top_slots) == 1 else None
            else:
                winner = None

            terminal_rewards = terminal_reward_vector(
                winner=winner,
                n_present=n_present,
                cfg=self.cfg,
                max_players=self._max_players,
            )

            initial_obs = steps[0][0]["observation"]
            prod_total = _total_production_capacity(initial_obs)
            denom_steps = max(1.0, float(getattr(self.cfg.env, "max_steps", 500)))

            # Build one asymmetric/player-perspective trajectory per *present*
            # replay player. Each assembler carries that player's owner-slot
            # mapping, home-planet state, comet/fleet radar, and source_mask.
            #
            # Curated players contribute policy (BC) labels. All other players
            # contribute value-only samples: their action labels are sentinel
            # ILLEGAL labels (target_idx == -1), which the legal-label gate in
            # ppo_core.ppo_update (trainable_flat = active_flat &
            # legal_label_flat & ~sentinel_flat) drops from the BC loss while
            # the value head still trains on every sample.
            curated_ids = {int(s) for s in allowed_slots if int(s) < n_present}
            for player_id in sorted(allowed_slots):
                if int(player_id) >= n_present:
                    self._last_epoch_skipped_player_turns += 1

            if not curated_ids:
                # Preserve existing behavior: games with no curated players
                # (here: none within the present-player range) are skipped
                # entirely, including their value-only perspectives.
                self._last_epoch_skipped_no_curated += 1
                continue

            assemblers: Dict[int, OrbitWarsAssembler] = {}
            tags: Dict[int, str] = {}
            nonce = secrets.token_hex(4)
            for player_id in range(n_present):
                asm = OrbitWarsAssembler(max_steps=500)
                asm.reset(initial_obs, player_id=int(player_id))
                tag = f"offline_{nonce}_p{int(player_id)}"
                assemblers[int(player_id)] = asm
                tags[int(player_id)] = tag
                self._traj[tag] = {"obs": [], "act": [], "rew": []}

            self._last_epoch_value_only_slots += n_present - len(curated_ids)

            # Play through the replay as one trajectory per curated player.
            # Preserve fleet_radar/known_comets across replay steps, but clear
            # only the step-local edge geometry each new frame. The cache is
            # shared across player perspectives so expensive raycasts are reused,
            # while each assembler still encodes owner slots/source rows relative
            # to its own player_id.
            shared_assembly_cache: Dict[str, Any] = {}
            for step_idx in range(len(steps) - 1):
                frame_data = steps[step_idx]
                next_frame_data = steps[step_idx + 1]

                if not frame_data:
                    continue

                obs_t = frame_data[0]["observation"]
                obs_next = next_frame_data[0]["observation"] if next_frame_data else obs_t
                planets = obs_t.get("planets", []) or []

                if shared_assembly_cache.get("step") != int(step_idx):
                    fleet_radar = shared_assembly_cache.get("fleet_radar", {})
                    known_comets = shared_assembly_cache.get("known_comets", set())
                    shared_assembly_cache.clear()
                    shared_assembly_cache["step"] = int(step_idx)
                    shared_assembly_cache["fleet_radar"] = fleet_radar
                    shared_assembly_cache["known_comets"] = known_comets

                # Extract comet geometry for raycasting historical orders.
                comet_info = {}
                for grp in (obs_t.get("comets") or []):
                    pids = grp.get("planet_ids", [])
                    paths = grp.get("paths", [])
                    pidx = int(grp.get("path_index", 0))
                    for b_idx, c_pid in enumerate(pids):
                        if b_idx < len(paths):
                            comet_info[int(c_pid)] = (paths[b_idx], pidx, len(paths[b_idx]))

                raw_next = _production_advantage_raw(
                    obs_next,
                    max_players=self._max_players,
                    n_agents=n_present,
                    total_prod=prod_total,
                )

                # Dense competitive production-advantage holding reward.
                #
                # Each step rewards the current normalized production advantage:
                #
                #   coef * ((self_prod - max_enemy_prod) / total_map_production) / max_steps
                #
                # Because raw_next is clipped to [-1, +1] and divided by max_steps,
                # the maximum undiscounted shaped reward sum over a full episode is
                # approximately +/- production_coef. With production_coef < terminal_win,
                # terminal outcome remains slightly more important than shaping.
                step_rew_vec = (
                    float(self.production_coef) * raw_next / denom_steps
                ).astype(np.float32)

                for player_id, assembler in assemblers.items():
                    curated = player_id in curated_ids
                    if curated and player_id >= len(next_frame_data):
                        self._last_epoch_skipped_player_turns += 1
                        continue

                    assembled = assembler.assemble(
                        obs_t,
                        step_idx,
                        compute_expert=False,
                        shared_cache=shared_assembly_cache,
                    )
                    obs_flat = pack_observation_np(assembled)

                    # Default per-source label is WAIT: target == self with
                    # OPT_SKIP. The option axis has width N_ACTION_OPTIONS == 2
                    # (OPT_SKIP == 0, OPT_MAX_SEND == 1); a launch is labelled as
                    # a full send below.
                    target_indices = np.arange(self._max_planets, dtype=np.int64)
                    option_indices = np.full(self._max_planets, OPT_SKIP, dtype=np.int64)

                    if not curated:
                        # Value-only perspective: sentinel ILLEGAL labels via
                        # target_idx == -1. With the full-send space the self
                        # node is the *legal* WAIT action, so an illegal label
                        # can no longer be encoded by an option; instead a
                        # negative target is dropped by ppo_update's trainable
                        # gate (sentinel_flat). The value head trains on the
                        # sample regardless. The order reverse-mapping (and its
                        # raycasts) is skipped entirely.
                        target_indices = np.full(self._max_planets, -1, dtype=np.int64)
                    else:
                        # Per-perspective action tensor. WAIT stays target=self.
                        # Launches from this replay player are reverse-mapped to
                        # full-send target labels below.
                        pass

                        player_action_block = next_frame_data[player_id]
                        raw_orders = player_action_block.get("action", []) or []

                        for order in raw_orders:
                            if len(order) < 3:
                                continue
                            src_pid = int(order[0])
                            raw_angle = float(order[1])
                            ships_sent = int(order[2])

                            src_idx = next(
                                (i for i, p in enumerate(planets) if int(p[0]) == src_pid),
                                None,
                            )
                            if src_idx is None or src_idx >= self._max_planets:
                                continue

                            # Only train labels for source rows that this
                            # perspective can actually act from. This protects
                            # against stale/invalid replay orders and keeps labels
                            # aligned with the assembled source_mask.
                            if float(assembled["source_mask"][src_idx]) <= 0.5:
                                continue

                            src_planet = planets[src_idx]
                            src_garrison = max(1, int(src_planet[5]))
                            send_frac = float(ships_sent) / float(src_garrison)

                            launch_x = float(src_planet[2]) + math.cos(raw_angle) * (float(src_planet[4]) + 0.1)
                            launch_y = float(src_planet[3]) + math.sin(raw_angle) * (float(src_planet[4]) + 0.1)

                            dummy_fleet = [
                                9999,
                                int(player_id),
                                launch_x,
                                launch_y,
                                raw_angle,
                                src_pid,
                                ships_sent,
                            ]
                            actual_hit_pid, _ = assembler._raycast_fleet(
                                dummy_fleet,
                                obs_t,
                                step_idx,
                                comet_info,
                            )

                            best_t_idx = -1
                            if actual_hit_pid is not None:
                                best_t_idx = next(
                                    (i for i, p in enumerate(planets) if int(p[0]) == int(actual_hit_pid)),
                                    -1,
                                )

                            if (
                                best_t_idx < 0
                                or best_t_idx >= len(planets)
                                or best_t_idx >= self._max_planets
                                or best_t_idx == src_idx
                            ):
                                # Leave canonical WAIT label for this source.
                                continue

                            # Submission action space: any launch is labelled as
                            # a full send (OPT_MAX_SEND). The send fraction is
                            # carried implicitly by the engine's order mapping.
                            target_indices[src_idx] = int(best_t_idx)
                            option_indices[src_idx] = OPT_MAX_SEND

                    act_pair = np.stack([target_indices, option_indices], axis=-1).astype(np.int64)

                    tag = tags[player_id]
                    self._traj[tag]["obs"].append(obs_flat)
                    self._traj[tag]["act"].append(act_pair)
                    self._traj[tag]["rew"].append(np.float32(step_rew_vec[player_id]))

            for player_id, tag in tags.items():
                term = float(terminal_rewards[player_id]) if player_id < terminal_rewards.shape[0] else 0.0
                self._finalize_imitation_trajectory(tag, MatchOutcome(term))

            self._matches_processed += 1
            processed_this_epoch += 1

            if self._matches_processed % 50 == 0:
                print(
                    f"[*] Offline Worker {self.worker_id}: "
                    f"Sent {self._matches_processed} complete matches to Learner."
                )

        self._last_epoch_processed = processed_this_epoch
        return processed_this_epoch

    def heartbeat(self):
        total_steps = sum(len(buf["obs"]) for buf in self._traj.values())
        return {
            "active_matches": 0,
            "learner_q_size": self.learner_client.q.qsize(),
            "traj_in_memory": len(self._traj),
            "ep_sem_value": -1,
            "total_steps_in_memory": total_steps,
            "offline_mode": True,
            "symmetric_full_state": False,
            "player_perspective": True,
            "matches_processed": self._matches_processed,
            "epochs_completed": self._epochs_completed,
            "last_epoch_found": self._last_epoch_found,
            "last_epoch_processed": self._last_epoch_processed,
            "last_epoch_curated_slots": self._last_epoch_curated_slots,
            "last_epoch_value_only_slots": self._last_epoch_value_only_slots,
            "last_epoch_skipped_no_curated": self._last_epoch_skipped_no_curated,
            "last_epoch_skipped_player_turns": self._last_epoch_skipped_player_turns,
        }
