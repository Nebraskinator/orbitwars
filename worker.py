"""
Distributed Rollout Worker for Orbit Wars.

Self-play roll-outs: each match hosts two policy agents (slot 0 and slot 1),
both controlled by the same OrbitTransformer. Per turn we:

  1. Assemble obs for every agent that needs to act (both agents per turn
     in self-play), pulling per-agent perspective via OrbitWarsAssembler.
  2. Stack the per-agent flat obs into one numpy batch and ship it to the
     central InferenceActor in a single Ray RPC.
  3. Route the returned (target, frac) per-planet actions back to their
     match+agent slot, convert via ``OrbitWarsAssembler.map_actions_to_orders``
     into engine orders ``[[from_id, angle, ships], ...]``, and step the
     Kaggle env.
  4. On terminal, finalize the trajectory (terminal-only reward for v1) and
     reset the match for the next episode.

Single-thread, synchronous: each worker steps its matches, calls the
InferenceActor for a batched action, and submits finished episodes.
"""

from __future__ import annotations

import logging
import queue
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Final
import numpy as np
import ray

from config import RunConfig
from orbit_obs_reftrace import OrbitWarsAssembler, schema_metadata

logger = logging.getLogger(__name__)

# DORA-style search is imported lazily inside the worker so that turning it off
# (cfg.search.enabled == False) keeps the module's import surface and behavior
# identical to the pre-search pipeline.


# ---------------------------------------------------------------------------
# SYNC LEARNER CLIENT (env-agnostic)
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
        obs: np.ndarray,   # [T, obs_dim]
        act: np.ndarray,   # [T, P, 2]
        logp: np.ndarray,  # [T, P] per-source behavior logp
        val: np.ndarray,   # [T]
        var: np.ndarray,   # [T]
        rew: np.ndarray,   # [T]
        done: np.ndarray,  # [T]
        sbr: Optional[np.ndarray] = None,       # [T, P, 2] SBR joint action target
        searched: Optional[np.ndarray] = None,  # [T] searched flag
    ) -> bool:
        # If the queue is full, evict the oldest trajectory.
        if self.q.full():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put_nowait((obs, act, logp, val, var, rew, done, sbr, searched))
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
        # SBR targets are present only when search ran for every episode in the
        # batch; otherwise pass None and the learner zero-fills (CE stays inert).
        if all(len(it) > 7 and it[7] is not None for it in items):
            sbr_cat = np.concatenate([it[7] for it in items], axis=0).astype(np.int64)
            searched_cat = np.concatenate([it[8] for it in items], axis=0).astype(np.float32)
        else:
            sbr_cat = None
            searched_cat = None
        return (
            np.concatenate([it[0] for it in items], axis=0),
            np.concatenate([it[1] for it in items], axis=0).astype(np.int64),
            np.concatenate([it[2] for it in items], axis=0).astype(np.float32),
            np.concatenate([it[3] for it in items], axis=0).astype(np.float32),
            np.concatenate([it[4] for it in items], axis=0).astype(np.float32),
            np.concatenate([it[5] for it in items], axis=0).astype(np.float32),
            np.concatenate([it[6] for it in items], axis=0).astype(np.float32),
            lengths,
            sbr_cat,
            searched_cat,
        )


# ---------------------------------------------------------------------------
# Per-agent terminal outcome handed back to RolloutWorker via "DONE" events.
# ---------------------------------------------------------------------------

@dataclass
class MatchOutcome:
    """Lightweight stand-in for the old ``battle`` object on terminal."""
    won: bool


def _production_totals_by_player(obs: Dict[str, Any], n_agents: int) -> List[int]:
    """Return raw production totals for each active player slot.

    Planet rows from the kaggle env are ``(pid, owner, x, y, radius, ships,
    prod)``. Owners outside ``[0, n_agents)`` are neutral/pad/irrelevant for
    the active match and are ignored.
    """
    totals = [0] * int(n_agents)
    for p in obs.get("planets", []) or []:
        owner = int(p[1])
        if 0 <= owner < int(n_agents):
            totals[owner] += int(p[6])
    return totals

def _total_production_capacity(obs: Dict[str, Any]) -> float:
    """Total positive production available on the map, including neutrals.

    This is intentionally map-level rather than controlled-production-only so
    the reward scale stays stable across early/mid/late game and matches the
    offline worker.
    """
    total = 0.0
    for p in obs.get("planets", []) or []:
        prod = float(p[6])
        if prod > 0.0:
            total += prod
    return max(1.0, float(total))


def _production_margin_scores(
    obs: Dict[str, Any],
    n_agents: int,
    total_prod: float,
) -> List[float]:
    """Per-player normalized production advantage scores.

    score_i = (prod_i - max(prod_j for j != i)) / total_map_production

    This is a bounded state reward signal in [-1, +1]. The per-step shaped
    reward should divide this by max_steps so the maximum possible shaped
    reward sum is approximately production_coef.
    """
    n_agents = int(n_agents)
    total_prod = max(1.0, float(total_prod))
    totals = _production_totals_by_player(obs, n_agents)

    scores: List[float] = []
    for i in range(n_agents):
        if n_agents <= 1:
            max_enemy = 0
        else:
            max_enemy = max(totals[j] for j in range(n_agents) if j != i)

        score = (float(totals[i]) - float(max_enemy)) / total_prod
        scores.append(float(np.clip(score, -1.0, 1.0)))

    return scores

def pack_observation_np(d: Dict[str, Any]) -> np.ndarray:
    """
    NumPy-only flat observation packer for rollout workers.

    Keeps the same flat layout as ppo_core.pack_observation:
      global, planet, edge, edge_mask, planet_mask, source_mask
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

def allocate_match_player_counts(
    n_matches: int, choices: Tuple[int, ...]
) -> List[int]:
    """Decide how many players each of ``n_matches`` matches will host.

    The allocation is *slot-balanced* across ``choices``: each choice ``c``
    gets approximately equal total player-slots. Concretely, for choices
    ``(c_1, ..., c_K)`` the goal is ``n_i * c_i ≈ n_j * c_j`` for all
    ``i, j`` while ``Σ n_i == n_matches``.

    For ``(2, 4)`` and ``n_matches`` divisible by 3 this yields exactly
    2/3 two-player matches and 1/3 four-player matches (slot count 50/50).
    Off-multiples round nearest with the residual placed in the largest
    bucket so total match count is preserved.

    Returns a list of length ``n_matches`` where each entry is a player
    count drawn from ``choices``.
    """
    if not choices:
        raise ValueError("choices must be non-empty")
    if n_matches < 0:
        raise ValueError("n_matches must be non-negative")
    if len(choices) == 1:
        return [int(choices[0])] * n_matches

    # Solve for per-choice match counts that equalize player-slots.
    #   slots_per_choice = total_slots / K        (target)
    #   n_i = slots_per_choice / c_i              (real-valued)
    #   total_slots = n_matches * K / Σ(1/c_i)
    K = len(choices)
    inv_sum = sum(1.0 / float(c) for c in choices)
    total_slots = (n_matches * K) / inv_sum
    target_slots_per_choice = total_slots / K

    counts = [int(round(target_slots_per_choice / float(c))) for c in choices]

    # Fix rounding so Σ counts == n_matches. Adjust the largest bucket.
    diff = n_matches - sum(counts)
    if diff != 0:
        idx_largest = max(range(K), key=lambda i: counts[i])
        counts[idx_largest] += diff
        # Guard against ``counts`` going negative on tiny n_matches.
        for i in range(K):
            if counts[i] < 0:
                counts[i] = 0
        # Re-fix if the negative-clamp threw the sum off again.
        diff = n_matches - sum(counts)
        if diff != 0:
            idx_largest = max(range(K), key=lambda i: counts[i])
            counts[idx_largest] += diff

    out: List[int] = []
    for n_i, c in zip(counts, choices):
        out.extend([int(c)] * n_i)
    return out


# ---------------------------------------------------------------------------
# Single match: one Kaggle Orbit Wars env + one assembler per agent.
# ---------------------------------------------------------------------------

class OrbitWarsMatch:
    """One self-play match between ``n_agents`` OrbitTransformer agents.

    ``n_agents`` is fixed for the lifetime of the match (2 for 1v1
    self-play, 4 for FFA). Each match owns one Kaggle env plus one
    ``OrbitWarsAssembler`` per slot. The assembler owns the per-episode
    orbital cache; we ``reset`` it from the initial obs once per episode.
    """

    def __init__(self, max_steps: int = 500, run_tag: str = "",
                 production_coef: float = 0.0,
                 n_agents: int = 2,
                 policy_ids: Optional[Sequence[int]] = None,
                 seed: Optional[int] = None):
        from kaggle_environments import make
        self._make_env = make
        self._base_seed = int(seed) if seed is not None else int(secrets.randbelow(2_000_000_000))
        self._episode_index = 0
        self.env = None

        self.max_steps = max_steps
        self.run_tag = run_tag
        # Coefficient on the per-step ``Δ(total production)`` shaping reward.
        # Held on the match (rather than read from cfg in step) so the env
        # wrapper stays the single source of dense reward bookkeeping.
        self.production_coef = float(production_coef)
        # Number of agent slots in this match (2 for 1v1, 4 for FFA).
        self.n_agents = int(n_agents)

        # Static policy assignment per slot. ``0`` means the current policy
        # (training); a positive int ``k`` means the InferenceActor routes
        # this slot through league snapshot ``k - 1``. Fixed for the lifetime
        # of the match -- the actor may rotate which checkpoint sits in slot
        # ``k`` over time, but the worker is unaware.
        if policy_ids is None:
            self.policy_ids: List[int] = [0] * self.n_agents
        else:
            if len(policy_ids) != self.n_agents:
                raise ValueError(
                    f"policy_ids length {len(policy_ids)} != n_agents {self.n_agents}"
                )
            self.policy_ids = [int(p) for p in policy_ids]

        self.assemblers = [OrbitWarsAssembler(max_steps=max_steps)
                           for _ in range(self.n_agents)]

        # Shared board-level assembly cache.
        # Persistent entries:
        #   fleet_radar, known_comets
        # Step-local entries:
        #   edge_geometry, edge_geometry_key
        self._shared_assembly_cache: Dict[str, Any] = {}

        # Tags rotate per episode -- the Learner relies on tag uniqueness for
        # per-trajectory storage in RolloutWorker._traj.
        self.tags: List[str] = [""] * self.n_agents

        # Latest raw kaggle obs per agent (used by the action interpreter to
        # look up live planet ownership / garrison / comet paths).
        self._raw_obs: List[Optional[Dict[str, Any]]] = [None] * self.n_agents

        # Most recent assembled obs that still needs an action; cleared after
        # the step that consumes it.
        self._pending: List[bool] = [False] * self.n_agents

        # Fixed per-episode map production capacity, including neutral planets.
        # Used to normalize the per-turn production-advantage holding reward.
        self._production_total_capacity: float = 1.0

        # Per-agent shaping reward emitted by the most recent ``step`` call.
        # Read by ``OrbitWarsVectorEnv`` to fan out REWARD events.
        self.last_step_rewards: List[float] = [0.0] * self.n_agents

        self.step_count = 0
        self.done = False
        # Slot index of the unique highest-reward player on terminal,
        # or None for draws / mid-game.
        self.winner: Optional[int] = None

        self.reset()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def _make_seeded_env(self):
        """Create a fresh Orbit Wars environment for the current episode seed."""
        episode_seed = int(self._base_seed + self._episode_index)

        try:
            return self._make_env(
                "orbit_wars",
                debug=False,
                configuration={"randomSeed": episode_seed},
            )
        except TypeError:
            env = self._make_env("orbit_wars", debug=False)
            try:
                env.configuration.randomSeed = episode_seed
            except Exception:
                pass
            return env
        
    def reset(self):
        self.step_count = 0
        self.done = False
        self.winner = None
        nonce = secrets.token_hex(4)
        self.tags = [f"{self.run_tag}_{nonce}_{i}" for i in range(self.n_agents)]

        # Critical: recreate the Kaggle env with a fresh deterministic seed per
        # episode. Otherwise concurrent matches can reuse the same default map.
        self.env = self._make_seeded_env()
        self._episode_index += 1
        
        self._shared_assembly_cache.clear()
        states = self.env.reset(num_agents=self.n_agents)
        for i, state in enumerate(states):
            obs = self._normalize_obs(state)
            self.assemblers[i].reset(obs, player_id=i)
            self._raw_obs[i] = obs
            self._pending[i] = True

        # Fix the reward normalization denominator for the episode. Include
        # neutral production so the scale is stable and the maximum possible
        # per-episode shaped reward is controlled by production_coef.
        baseline_obs = self._raw_obs[0] if self._raw_obs and self._raw_obs[0] is not None else {}
        self._production_total_capacity = _total_production_capacity(baseline_obs)
        self.last_step_rewards = [0.0] * self.n_agents

    @staticmethod
    def _normalize_obs(state) -> Dict[str, Any]:
        """Coerce a kaggle agent state into a plain dict."""
        obs = getattr(state, "observation", state)
        if hasattr(obs, "to_dict"):
            obs = obs.to_dict()
        if not isinstance(obs, dict):
            obs = {k: getattr(obs, k) for k in dir(obs) if not k.startswith("_")}
        return obs

    # ------------------------------------------------------------------
    # Obs assembly + stepping
    # ------------------------------------------------------------------

    def assemble_pending(self, compute_expert: bool = False) -> List[Tuple[int, str, np.ndarray, Optional[np.ndarray]]]:
        """For each agent slot whose obs hasn't been consumed yet, run the
        assembler and pack into a flat numpy float32 vector. Returns
        ``(slot, tag, obs_flat, expert_action)`` tuples.
        """
        out: List[Tuple[int, str, np.ndarray, Optional[np.ndarray]]] = []

        shared_cache = self._shared_assembly_cache

        # Keep persistent fleet projection cache across steps, but only keep
        # edge geometry for the current board step.
        if shared_cache.get("step") != int(self.step_count):
            fleet_radar = shared_cache.get("fleet_radar", {})
            known_comets = shared_cache.get("known_comets", set())

            shared_cache.clear()
            shared_cache["step"] = int(self.step_count)
            shared_cache["fleet_radar"] = fleet_radar
            shared_cache["known_comets"] = known_comets

        for i in range(self.n_agents):
            if not self._pending[i]:
                continue
            d = self.assemblers[i].assemble(
                self._raw_obs[i],
                self.step_count,
                compute_expert=compute_expert,
                shared_cache=shared_cache,
            )
            obs_flat = pack_observation_np(d)
            out.append((i, self.tags[i], obs_flat, d.get("expert_action")))
        return out

    def step(self, act_per_agent: List[np.ndarray],
             raw_orders: Optional[Dict[int, List]] = None) -> bool:
        """Step the env with one ``[MAX_PLANETS, 2]`` action per agent.

        ``raw_orders`` optionally maps a slot index to pre-computed engine
        orders for that slot (e.g. produced by an external black-box agent
        such as a packaged submission). Slots present in ``raw_orders`` skip
        the host-side ``map_actions_to_orders`` conversion entirely; their
        entry in ``act_per_agent`` is ignored.

        Returns True if the episode ended on this step.
        """
        if self.done:
            return True

        engine_orders: List[List[List]] = []
        for i, act_pair in enumerate(act_per_agent):
            if raw_orders is not None and i in raw_orders:
                engine_orders.append(raw_orders[i] or [])
                continue
            tgt = np.asarray(act_pair[..., 0], dtype=np.int64)
            frc = np.asarray(act_pair[..., 1], dtype=np.int64)
            orders = self.assemblers[i].map_actions_to_orders(
                tgt,
                frc,
                self._raw_obs[i],
                step=self.step_count,
                allow_exact_fallback=False,
            )
            if orders is None:
                orders = []
            engine_orders.append(orders)

        states = self.env.step(engine_orders)
        self.step_count += 1

        for i, state in enumerate(states):
            self._raw_obs[i] = self._normalize_obs(state)

        # Dense competitive production-advantage holding reward.
        #
        # Each step rewards the current normalized production advantage:
        #
        #   coef * ((self_prod - max_enemy_prod) / total_map_production) / max_steps
        #
        # Because the score is clipped to [-1, +1] and divided by max_steps,
        # the maximum undiscounted shaped reward sum over a full episode is
        # approximately +/- production_coef. With production_coef < terminal_win,
        # terminal outcome remains slightly more important than shaping.
        coef = float(self.production_coef)
        denom_steps = max(1.0, float(self.max_steps))
        score_obs = self._raw_obs[0] if self._raw_obs and self._raw_obs[0] is not None else {}
        prod_scores_now = _production_margin_scores(
            score_obs,
            self.n_agents,
            self._production_total_capacity,
        )
        for i in range(self.n_agents):
            self.last_step_rewards[i] = coef * float(prod_scores_now[i]) / denom_steps

        statuses = [getattr(s, "status", "ACTIVE") for s in states]
        rewards = [getattr(s, "reward", 0) or 0 for s in states]

        episode_over = (
            all(st == "DONE" for st in statuses)
            or any(st in ["INVALID", "ERROR", "TIMEOUT"] for st in statuses)
            or self.step_count >= self.max_steps
        )

        if episode_over:
            self.done = True
            # Winner = unique slot with the strictly-highest terminal reward.
            # Ties (incl. all-equal) → draw, signalled by ``winner = None``.
            # Generalizes the old 2p ``rewards[0] > rewards[1]`` check to FFA.
            top = max(rewards)
            top_slots = [i for i, r in enumerate(rewards) if r == top]
            self.winner = top_slots[0] if len(top_slots) == 1 else None
            self._pending = [False] * self.n_agents
        else:
            self._pending = [True] * self.n_agents

        return self.done


# ---------------------------------------------------------------------------
# Vector wrapper: aggregate the pending obs across all matches into one batch
# every step so the InferenceActor sees a single fat call per turn.
# ---------------------------------------------------------------------------

class OrbitWarsVectorEnv:
    """Aggregates N Orbit Wars matches and exposes the same event-stream
    contract that ``RolloutWorker._run_rl_loop`` already consumes.

    Events emitted per ``step`` call:
      ``("STEP", tag, obs_flat, None)``        -- agent ``tag`` is awaiting
                                                  an action from the policy
      ``("DONE", tag, None, MatchOutcome)``    -- agent ``tag``'s episode
                                                  ended; ``MatchOutcome.won``
                                                  is True iff this slot won
    """

    def __init__(
        self,
        cfg: RunConfig,
        n_matches: int,
        run_tag: str = "",
        worker_id: int = 0,
        seed_base: int = 12345,
    ):
        self.cfg = cfg
        self.n_matches = int(n_matches)
        self.worker_id = int(worker_id)
        self.seed_base = int(seed_base)

        max_steps = int(getattr(cfg.env, "max_steps", 500))
        production_coef = float(getattr(cfg.reward, "production_coef", 0.05))

        # Resolve per-match player counts. ``cfg.env.n_agents`` is a tuple
        # of allowed counts; the allocator splits this worker's matches
        # across the choices so that *player slot* count is balanced. With
        # the default ``(2, 4)`` and an n_matches divisible by 3, that's
        # exactly 2/3 two-player matches and 1/3 four-player matches
        # (50/50 by player slot count).
        choices = getattr(cfg.env, "n_agents", (2,))
        if isinstance(choices, int):
            choices = (choices,)
        choices = tuple(int(c) for c in choices)
        per_match_n = allocate_match_player_counts(self.n_matches, choices)

        # Static league assignment per (match, slot). Rolled once at
        # construction and never rebound for the lifetime of the worker;
        # the InferenceActor can rotate which checkpoint sits behind a
        # given slot id, but this worker doesn't track that.
        per_match_policy_ids = self._roll_league_assignment(per_match_n)

        self.matches: List[OrbitWarsMatch] = [
            OrbitWarsMatch(
                max_steps=max_steps,
                run_tag=f"{run_tag}_m{m_idx}",
                production_coef=production_coef,
                n_agents=n,
                policy_ids=pids,
                seed=(
                    self.seed_base
                    + self.worker_id * 1_000_003
                    + m_idx * 10_007
                ),
            )
            for m_idx, (n, pids) in enumerate(zip(per_match_n, per_match_policy_ids))
        ]
        # Legacy attribute the worker reads in heartbeat (best-effort).
        self.active_pairs: List[Tuple[Any, Any]] = []
        self.ep_sem = None  # unused; kept for API parity.

    def _roll_league_assignment(
        self, per_match_n: Sequence[int]
    ) -> List[List[int]]:
        """Decide per-(match, slot) league policy_id assignment.

        - Each slot independently goes league with probability ``fraction``.
        - When league, the slot picks a uniform random index in
          ``[1, n_slots]`` (the InferenceActor maps that to a resident
          checkpoint).
        - Invariant: every match keeps at least one current-policy agent so
          the worker always produces training data for it.
        """
        league = getattr(self.cfg, "league", None)
        enabled = bool(getattr(league, "enabled", False)) if league else False
        n_slots = int(getattr(league, "n_slots", 0)) if league else 0
        if not enabled or n_slots <= 0:
            return [[0] * int(n) for n in per_match_n]

        # Skip league entirely when the worker isn't running PPO -- imitation
        # has its own action path and doesn't call infer_batch.
        mode = str(getattr(self.cfg.learner, "mode", "ppo"))
        if mode.startswith("imitation"):
            return [[0] * int(n) for n in per_match_n]

        import random as _rand
        rng = _rand.Random()
        fraction = float(league.fraction)

        out: List[List[int]] = []
        for n in per_match_n:
            n = int(n)
            slots = [0] * n
            for i in range(n):
                if rng.random() < fraction:
                    slots[i] = 1 + rng.randrange(n_slots)
            # Training-data invariant: keep ≥1 current per match.
            if n > 0 and all(s != 0 for s in slots):
                slots[rng.randrange(n)] = 0
            out.append(slots)
        return out

    def step(
        self, actions_dict: Dict[str, np.ndarray], compute_expert: bool=False
    ) -> List[Tuple[str, str, Optional[np.ndarray], Optional[MatchOutcome]]]:
        events: List[Tuple] = []

        for match in self.matches:
            # Snapshot the pre-reset tags so REWARD/DONE events attach to the
            # episode that just ran, regardless of player count or whether
            # ``match.reset()`` rotates the tags below.
            current_tags = list(match.tags)

            # Step the match only when *all* its agents have actions queued.
            if (not match.done
                    and all(t in actions_dict for t in current_tags)):
                act_per_agent = [actions_dict[t] for t in current_tags]
                terminal = match.step(act_per_agent)
                # Always emit per-step production-advantage holding rewards for
                # the action just consumed -- one REWARD per slot.
                step_rewards = match.last_step_rewards
                for i, tag in enumerate(current_tags):
                    events.append(("REWARD", tag, step_rewards[i], None))
                if terminal:
                    # Episode just terminated -- emit DONE for every slot.
                    # ``match.winner`` is the unique top-reward slot or
                    # ``None`` for draws (see ``OrbitWarsMatch.step``).
                    for i, tag in enumerate(current_tags):
                        won = (match.winner == i)
                        events.append(("DONE", tag, None, MatchOutcome(won=won)))
                    match.reset()  # rotates tags, re-pends obs for next ep.

            # Emit STEP events for any agent slot whose obs still needs to be
            # converted into an action.
            for slot, tag, obs_flat, expert_action in match.assemble_pending(compute_expert=compute_expert):
                events.append((
                    "STEP",
                    tag,
                    obs_flat,
                    {
                        "match": match,
                        "slot": int(slot),
                        "expert_action": expert_action,
                        "policy_id": int(match.policy_ids[slot]),
                    },
                ))
            
                # Mark assembled so the same obs isn't enqueued twice.
                slot_idx = match.tags.index(tag)
                match._pending[slot_idx] = False

        return events

    def release_slot(self):
        # No-op kept for API parity with RolloutWorker._finalize_*.
        return


# ---------------------------------------------------------------------------
# ROLLOUT WORKER (RAY ACTOR)
# ---------------------------------------------------------------------------

class RolloutWorker:
    def __init__(self, cfg: RunConfig, inference_actor, learner_actor,
                 pairs: int, server_port: int = 0,
                 worker_id: int = 0,
                 seed_base: int = 12345,
                 search_actor=None):
        self.cfg = cfg
        self.worker_id = int(worker_id)
        self.seed_base = int(seed_base)

        self.learner_client = SyncLearnerClient(learner_actor, cfg)
        self.inference_actor = inference_actor
        # Dedicated GPU actor that owns the SBR forward model (decode -> sim ->
        # assemble -> value). Only used when cfg.search.enabled. None otherwise.
        self.search_actor = search_actor
        # ``server_port`` is accepted but unused (the env is in-process).
        self.vec_env = OrbitWarsVectorEnv(
            cfg,
            n_matches=int(pairs),
            run_tag=f"w{self.worker_id}",
            worker_id=self.worker_id,
            seed_base=self.seed_base,
        )
        self._traj: Dict[str, Dict[str, list]] = {}
        self._meta = schema_metadata()
        self._max_planets = int(self._meta["max_planets"])
        
        self.mode = str(getattr(cfg.learner, "mode", "ppo"))
        self.use_heuristic_expert = self.mode.startswith("imitation")

        # DORA-style 1-ply equilibrium search. Off by default; only active for
        # PPO rollouts (not imitation). When off, none of the search code runs.
        self.search_cfg = getattr(cfg, "search", None)
        self.use_search = (
            bool(getattr(self.search_cfg, "enabled", False))
            and not self.use_heuristic_expert
            and (self.search_actor is not None)
        )
        self._search_sim = None  # legacy CPU sim handle (unused; SearchActor owns the model)
        
        # Keep imitation chunks short enough to avoid memory blowups, but long enough
        # that JEPA sees real adjacent same-tag transitions.
        self.imitation_chunk_len = int(getattr(cfg.rollout, "imitation_chunk_len", 32))

    def run(self):
        """Synchronous loop: step env -> infer batch -> route actions back."""
        self._run_rl_loop()
        
    def _submit_imitation_sequence(
        self,
        obs_list: list,
        act_list: list,
        rew_list: list,
        done_last: bool,
    ):
        """
        Submit one ordered same-tag imitation sequence.
    
        This preserves:
          - BC labels from heuristic actions
          - reward targets for value loss
          - adjacent rows for JEPA t -> t+1
        """
        T = len(obs_list)
        if T <= 0:
            return
    
        obs_stacked = np.ascontiguousarray(
            np.stack(obs_list, axis=0),
            dtype=np.float32,
        )
        act_stacked = np.ascontiguousarray(
            np.stack(act_list, axis=0),
            dtype=np.int64,
        )
    
        rewards = np.asarray(rew_list, dtype=np.float32)
        dones = np.zeros(T, dtype=np.float32)
        if done_last:
            dones[-1] = 1.0
    
        P = int(act_stacked.shape[1])
        zeros_step = np.zeros(T, dtype=np.float32)
        zeros_logp = np.zeros((T, P), dtype=np.float32)
    
        self.learner_client.submit_episode(
            obs_stacked,
            act_stacked,
            zeros_logp,          # logp old [T, P], unused in imitation
            zeros_step.copy(),   # value old; GAE uses this as baseline
            zeros_step.copy(),   # variance old
            rewards,
            dones,
        )
    
    
    def _flush_imitation_ready_chunks(self):
        """
        Flush bounded nonterminal same-tag chunks.
    
        Only flush steps that already have rewards. Usually len(obs/act) is either
        equal to len(rew), or one larger because the latest action is waiting for
        next step's reward.
        """
        if not self.use_heuristic_expert:
            return
    
        chunk_len = int(self.imitation_chunk_len)
        if chunk_len <= 1:
            chunk_len = 2
    
        for tag, buf in list(self._traj.items()):
            ready = min(len(buf["obs"]), len(buf["act"]), len(buf["rew"]))
    
            while ready >= chunk_len:
                n = chunk_len
    
                self._submit_imitation_sequence(
                    obs_list=buf["obs"][:n],
                    act_list=buf["act"][:n],
                    rew_list=buf["rew"][:n],
                    done_last=False,
                )
    
                del buf["obs"][:n]
                del buf["act"][:n]
                del buf["rew"][:n]
    
                # logp/val/var are not used in imitation, but keep buffers aligned
                # if they exist.
                if "logp" in buf:
                    del buf["logp"][:min(n, len(buf["logp"]))]
                if "val" in buf:
                    del buf["val"][:min(n, len(buf["val"]))]
                if "var" in buf:
                    del buf["var"][:min(n, len(buf["var"]))]
    
                ready = min(len(buf["obs"]), len(buf["act"]), len(buf["rew"]))
    
    
    def _finalize_imitation_trajectory(self, tag: str, outcome: MatchOutcome):
        """
        Add terminal reward and flush all remaining rewarded imitation steps.
        """
        buf = self._traj.pop(tag, None)
        if buf is None or not buf.get("act"):
            self.vec_env.release_slot()
            return
    
        ready = min(len(buf["obs"]), len(buf["act"]), len(buf["rew"]))
        if ready <= 0:
            self.vec_env.release_slot()
            return
    
        terminal_reward = (
            self.cfg.reward.terminal_win if outcome.won
            else self.cfg.reward.terminal_loss
        )
    
        # Add terminal outcome to the final rewarded action.
        buf["rew"][ready - 1] = float(buf["rew"][ready - 1]) + float(terminal_reward)
    
        self._submit_imitation_sequence(
            obs_list=buf["obs"][:ready],
            act_list=buf["act"][:ready],
            rew_list=buf["rew"][:ready],
            done_last=True,
        )
    
        self.vec_env.release_slot()

    def _run_rl_loop(self):
        actions_dict: Dict[str, np.ndarray] = {}

        while True:
            events = self.vec_env.step(actions_dict, compute_expert=self.use_heuristic_expert)
            actions_dict = {}

            step_tags: List[str] = []
            step_obs: List[np.ndarray] = []
            step_extra: List[dict] = []

            done_events = []

            for event_type, tag, payload, extra in events:
                if event_type == "STEP":
                    step_tags.append(tag)
                    step_obs.append(payload)
                    step_extra.append(extra)
            
                elif event_type == "REWARD":
                    # Per-step shaping reward attaches to the most recent action.
                    traj = self._traj.get(tag)
                    if traj is not None:
                        traj["rew"].append(float(payload))
            
                elif event_type == "DONE":
                    # Delay finalization until after all REWARD events in this env step
                    # have been attached.
                    done_events.append((tag, extra))
            
            # Finalize terminal tags after processing all rewards from this env step.
            for tag, outcome in done_events:
                if self.use_heuristic_expert:
                    self._finalize_imitation_trajectory(tag, outcome)
                else:
                    self._finalize_trajectory(tag, outcome)
            
            # In imitation mode, flush bounded nonterminal chunks only after rewards/DONE
            # have been processed. This avoids losing terminal rewards.
            if self.use_heuristic_expert:
                self._flush_imitation_ready_chunks()

            if step_obs:
                P = self._max_planets
            
                if self.use_heuristic_expert:
                    N = len(step_obs)
            
                    target_act = np.zeros((N, P), dtype=np.int64)
                    frac_act = np.zeros((N, P), dtype=np.int64)
            
                    for j, info in enumerate(step_extra):
                        act_pair = info["expert_action"]
            
                        target_act[j] = act_pair[:, 0].astype(np.int64)
                        frac_act[j] = act_pair[:, 1].astype(np.int64)
            
                    for i, tag in enumerate(step_tags):
                        act_pair = np.stack(
                            [target_act[i], frac_act[i]], axis=-1
                        ).astype(np.int64)
            
                        actions_dict[tag] = act_pair
            
                        traj = self._traj.setdefault(
                            tag,
                            {"obs": [], "act": [], "logp": [], "val": [], "var": [], "rew": []},
                        )
            
                        # Store obs/action now. Reward arrives on the next env step.
                        traj["obs"].append(step_obs[i])
                        traj["act"].append(act_pair)
            
                else:
                    obs_batch = np.ascontiguousarray(
                        np.stack(step_obs, axis=0),
                        dtype=np.float32,
                    )
                    policy_ids = np.fromiter(
                        (int(info.get("policy_id", 0)) for info in step_extra),
                        dtype=np.int32,
                        count=len(step_extra),
                    )

                    try:
                        target_act, frac_act, logps, vals, var = ray.get(
                            self.inference_actor.infer_batch.remote(
                                step_tags, obs_batch, policy_ids
                            )
                        )
                    except Exception as e:
                        logger.error(f"Inference Actor call failed: {e}")
                        N = len(step_obs)
                        target_act = np.zeros((N, P), dtype=np.int64)
                        frac_act = np.zeros((N, P), dtype=np.int64)
                        logps = np.zeros((N, P), dtype=np.float32)
                        vals = np.zeros(N, dtype=np.float32)
                        var = np.zeros(N, dtype=np.float32)

                    # DORA search: compute the SBR joint-action *target* for
                    # gated current-policy rows. Does NOT change the executed
                    # action. No-op (None) when search is disabled.
                    search_sbr = None
                    search_searched = None
                    if self.use_search:
                        try:
                            search_sbr, search_searched = self._apply_search(
                                step_tags, step_obs, step_extra,
                                target_act, frac_act, logps, var,
                            )
                        except Exception as e:
                            logger.error(f"DORA search failed; no SBR targets: {e}")
                            search_sbr = search_searched = None

                    for i, tag in enumerate(step_tags):
                        act_pair = np.stack(
                            [target_act[i], frac_act[i]], axis=-1
                        )  # [P, 2]
                        actions_dict[tag] = act_pair

                        # League agents are opponents only: skip trajectory
                        # bookkeeping. Their REWARD/DONE events naturally
                        # no-op since the tag never enters ``self._traj``.
                        if int(policy_ids[i]) != 0:
                            continue

                        traj = self._traj.setdefault(
                            tag,
                            {"obs": [], "act": [], "logp": [], "val": [], "var": [],
                             "rew": [], "sbr": [], "searched": []},
                        )
                        traj["obs"].append(step_obs[i])
                        traj["act"].append(act_pair)
                        # Store the joint behavior logp (sum of per-source
                        # logps over active sources) as a per-step scalar, which
                        # is what ppo_update's clipped ratio expects.
                        traj["logp"].append(
                            np.float32(np.asarray(logps[i], dtype=np.float32).sum())
                        )
                        traj["val"].append(vals[i])
                        traj["var"].append(var[i])
                        # SBR target + searched flag -- only stored when search
                        # is active, so the search-off path ships nothing extra.
                        if self.use_search:
                            if search_sbr is not None:
                                traj["sbr"].append(np.asarray(search_sbr[i], dtype=np.int64))
                                traj["searched"].append(float(search_searched[i]))
                            else:
                                traj["sbr"].append(act_pair.astype(np.int64))
                                traj["searched"].append(0.0)
                del step_extra 
                del step_tags
                del step_obs

    def _finalize_trajectory(self, tag: str, outcome: MatchOutcome):
        """Computes terminal reward and ships the trajectory to the learner."""
        self.inference_actor.clear_cache.remote(tag)  # no-op for Markovian, kept for API parity

        if tag not in self._traj:
            return
        buf = self._traj.pop(tag)
        if not buf["act"]:
            self.vec_env.release_slot()
            return

        T = len(buf["act"])
        obs_stacked = np.stack(buf["obs"], axis=0)
        act_stacked = np.stack(buf["act"], axis=0).astype(np.int64)  # [T, P, 2]

        terminal_reward = (
            self.cfg.reward.terminal_win if outcome.won
            else self.cfg.reward.terminal_loss
        )

        # Dense per-step production-advantage holding rewards -- one entry per
        # action the agent took.. If an episode terminates without any step
        # rewards arriving (e.g. INVALID before vec_env emits REWARD), pad
        # with zeros so length matches ``T``.
        step_rew = buf["rew"]
        rewards = np.zeros(T, dtype=np.float32)
        if step_rew:
            n = min(T, len(step_rew))
            rewards[:n] = np.asarray(step_rew[:n], dtype=np.float32)
        rewards[-1] += terminal_reward
        dones = np.zeros(T, dtype=np.float32)
        dones[-1] = 1.0

        logp_stacked = np.ascontiguousarray(
            np.stack(buf["logp"], axis=0),
            dtype=np.float32,
        )  # [T, P]

        # SBR search targets, if this trajectory carried them.
        sbr_stacked = None
        searched_stacked = None
        if buf.get("sbr"):
            n = min(T, len(buf["sbr"]))
            sbr_stacked = np.stack(buf["sbr"][:n], axis=0).astype(np.int64)  # [T, P, 2]
            searched_stacked = np.asarray(buf["searched"][:n], dtype=np.float32)
            if n < T:  # pad to T if the last action lacked a stored target
                pad = T - n
                sbr_stacked = np.concatenate(
                    [sbr_stacked, act_stacked[n:].astype(np.int64)], axis=0)
                searched_stacked = np.concatenate(
                    [searched_stacked, np.zeros(pad, dtype=np.float32)], axis=0)

        self.learner_client.submit_episode(
            obs_stacked,
            act_stacked,
            logp_stacked,
            np.asarray(buf["val"], dtype=np.float32),
            np.asarray(buf["var"], dtype=np.float32),
            rewards,
            dones,
            sbr=sbr_stacked,
            searched=searched_stacked,
        )
        self.vec_env.release_slot()

    # ------------------------------------------------------------------
    # 1-ply sampled best-response search (active only when use_search).
    # ------------------------------------------------------------------
    def _apply_search(self, step_tags, step_obs, step_extra,
                      target_act, frac_act, logps, var):
        """Compute the sampled-best-response (SBR) joint-action *target* for the
        current-policy slots, using one shared K x K sim grid per match.

        The executed rollout action is NOT changed -- the PPO action stays.
        Search only stores a separate distillation target. Per match we sample
        K = max(n_self, n_opp) candidates for each player ONCE, build a single
        K x K grid, simulate it once, and read a best-response off it for every
        current-policy slot (player 0 over rows, player 1 over columns) -- so we
        never re-simulate as we cycle through players.

        2-player matches only for v1. Caught by the caller on failure.

        Returns
        -------
        sbr_act  : int64 ``[N, P, 2]`` per-row SBR joint action (defaults to the
                   executed PPO action on non-searched rows -- ignored there).
        searched : float32 ``[N]`` 1.0 on rows where search produced a target.
        """
        import search as _search

        sc = self.search_cfg
        if int(getattr(sc, "depth", 1)) != 1:
            raise NotImplementedError("search.depth > 1 not yet implemented")
        if sc.gate_mode == "off":
            N = len(step_tags)
            return (np.stack([target_act, frac_act], axis=-1).astype(np.int64),
                    np.zeros(N, dtype=np.float32))

        K = max(int(sc.n_self_samples), int(sc.n_opp_samples))
        gamma = float(sc.gamma)
        aggregation = str(getattr(sc, "aggregation", "mean"))
        N = len(step_tags)

        sbr_act = np.stack([target_act, frac_act], axis=-1).astype(np.int64)  # [N,P,2]
        searched = np.zeros(N, dtype=np.float32)

        # Group batch rows by match; need both slots present to build the grid.
        by_match: Dict[int, Dict[int, int]] = {}
        for j, info in enumerate(step_extra):
            by_match.setdefault(id(info["match"]), {})[int(info["slot"])] = j

        # Eligible matches (2p, both slots in this step) and which slots to score.
        matches: List[Tuple[Any, Dict[int, int], List[int]]] = []
        for slots in by_match.values():
            if 0 not in slots or 1 not in slots:
                continue
            match = step_extra[slots[0]]["match"]
            if int(match.n_agents) != 2:
                continue
            score_slots = [p for p in (0, 1)
                           if int(step_extra[slots[p]]["policy_id"]) == 0]
            if score_slots:
                matches.append((match, slots, score_slots))

        if not matches:
            return sbr_act, searched

        # Optional global var gating across all scoreable rows.
        if sc.gate_mode == "var_topk":
            scoreable = [slots[p] for (_m, slots, sslots) in matches for p in sslots]
            vs = np.array([var[r] for r in scoreable], dtype=np.float32)
            if vs.size:
                kkeep = max(1, int(round(float(sc.gate_fraction) * vs.size)))
                thresh = np.sort(vs)[::-1][kkeep - 1]
                matches = [
                    (m, slots, [p for p in sslots if var[slots[p]] >= thresh])
                    for (m, slots, sslots) in matches
                ]
                matches = [(m, slots, sslots) for (m, slots, sslots) in matches if sslots]
        if not matches:
            return sbr_act, searched

        # --- Hand the surviving (match, scored-slot) set to the GPU SearchActor.
        #     It samples K candidates/player, fuses every (self, opp) candidate
        #     pair into one on-device sim -> assemble -> value pipeline, and
        #     returns each scored slot's SBR joint action. The old CPU sim/assembly
        #     path (simulate_match_grid + per-slot assemble_batch + evaluate_values)
        #     is gone -- the SearchActor owns the whole forward model now.
        prop_obs: List[np.ndarray] = []
        base_obs_list: List[Dict[str, Any]] = []
        match_meta: List[Dict[str, Any]] = []
        for mi, (match, slots, sslots) in enumerate(matches):
            prop_obs.append(step_obs[slots[0]])
            prop_obs.append(step_obs[slots[1]])
            base_obs_list.append(match._raw_obs[0])
            match_meta.append({
                "step": int(match.step_count),
                "max_steps": int(match.max_steps),
                "n_agents": int(match.n_agents),
                "production_coef": float(match.production_coef),
                "prod_cap": float(match._production_total_capacity),
                "score_slots": list(sslots),
                "rows": {0: int(slots[0]), 1: int(slots[1])},
            })

        sbr_rows = ray.get(self.search_actor.search.remote(
            np.stack(prop_obs, axis=0), base_obs_list, match_meta))

        for row, payload in sbr_rows.items():
            tgt_list, frc_list = payload
            r = int(row)
            sbr_act[r, :, 0] = np.asarray(tgt_list, dtype=np.int64)
            sbr_act[r, :, 1] = np.asarray(frc_list, dtype=np.int64)
            searched[r] = 1.0

        return sbr_act, searched

    def heartbeat(self):
        """Telemetry endpoint."""
        total_steps_in_memory = sum(len(buf["obs"]) for buf in self._traj.values())
        league_agents = sum(
            sum(1 for p in m.policy_ids if p != 0)
            for m in self.vec_env.matches
        )
        return {
            "active_matches": self.vec_env.n_matches,
            "learner_q_size": self.learner_client.q.qsize(),
            "traj_in_memory": len(self._traj),
            "ep_sem_value": -1,  # unused legacy field
            "total_steps_in_memory": total_steps_in_memory,
            "league_agents": league_agents,
        }
