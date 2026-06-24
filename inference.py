"""
Inference Server and Model Management for Distributed RL.

This module provides the InferenceActor, which handles large-batch GPU forward
passes
"""

from __future__ import annotations
import math
import random
import time
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Final

import numpy as np
import ray
import torch
import torch.nn as nn
from config import RunConfig
import ppo_core
from ppo_core import masked_sample
from orbit_obs_reftrace import schema_metadata
import torch.nn.functional as F

# Setup logging
logger = logging.getLogger(__name__)

# Constants
CKPT_UPDATE_PATTERN: Final[re.Pattern] = re.compile(r"learner_update_(\d+)\.pt$")

# -----------------------------------------------------------------------------
# CHECKPOINT UTILITIES
# -----------------------------------------------------------------------------

def _extract_state_dict(ckpt: dict) -> dict:
    """
    Surgically extracts the model state dictionary from various checkpoint formats.

    Args:
        ckpt: A dictionary loaded via torch.load.

    Returns:
        dict: The raw state_dict containing model parameters.

    Raises:
        RuntimeError: If no valid state_dict structure is detected.
    """
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Expected dict from checkpoint, got {type(ckpt)}")

    # Check common wrapper keys used in RL frameworks
    for k in ("model", "state_dict", "net", "policy", "actor_critic"):
        v = ckpt.get(k)
        if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
            return v

    # If no wrapper, check if the dict itself is the state_dict
    if any(isinstance(x, torch.Tensor) for x in ckpt.values()):
        return ckpt

    raise RuntimeError("Could not find a model state_dict inside the checkpoint file.")
@dataclass
class InferenceStats:
    """Container for monitoring inference performance and throughput."""
    flushes: int = 0
    total_requests: int = 0
    avg_batch_size: float = 0.0


@dataclass
class _LeagueSlot:
    """One resident league checkpoint slot owned by the InferenceActor.

    ``model`` is a long-lived ``OrbitTransformer`` instance on the inference
    device. ``update_num`` is the checkpoint number currently loaded into
    that instance, or ``None`` if the slot has never been filled.
    """
    model: nn.Module
    update_num: Optional[int] = None

# -----------------------------------------------------------------------------
# ACTORS
# -----------------------------------------------------------------------------

@ray.remote
class WeightStore:
    def __init__(self):
        self.weights = None
        self.version = -1
        self.temp = 1.0

    def update(self, weights, version):
        self.weights = weights
        self.version = version

    def get_version(self):
        return self.version

    def get_weights(self):
        return self.weights

    def set_temp(self, temp: float):
        self.temp = float(temp)

    def get_temp(self):
        return self.temp

    def get_meta(self):
        return self.version, self.temp

class InferenceActor:
    """
    High-throughput inference server responsible for GPU kernel execution.

    Supports 'Snapshot Inference'—running multiple different model versions
    within the same batch by grouping observations by their assigned policy_id.
    """
    def __init__(self, cfg: RunConfig, weight_store: ray.actor.ActorHandle):
        self.cfg = cfg
        self.weight_store = weight_store
        self.device: Final[str] = str(cfg.infer.device)

        self.current_version = -1
        self.current_temp = cfg.learner.temp_start
        self.current_behavior_eps = float(
            getattr(cfg.learner, "behavior_eps", getattr(cfg.learner, "explore_eps", 0.0))
        )
        self.stats = InferenceStats()

        # Build the OrbitTransformer for the current policy.
        self.net = self.cfg.make_model().to(self.device).eval()
        self.net.enable_bf16_recurrent_path()
        self._last_sync_time = None

        # Schema-derived shapes for sampling.
        self.meta = schema_metadata()
        self.max_planets = int(self.meta["max_planets"])
        self.n_options = int(self.meta.get("n_action_options", self.meta["n_option_buckets"]))
        self._src_start, self._src_end = self.net.unpacker.offsets["source_mask"]

        # ------------------------------------------------------------------
        # League play state. Workers send ``policy_ids`` per row; id 0 is the
        # current policy (``self.net``), ids 1..n_slots route to a resident
        # league slot. Slot→checkpoint binding is owned here and rotates on
        # ``cfg.league.refresh_every_sec``.
        # ------------------------------------------------------------------
        self.league_cfg = getattr(cfg, "league", None)
        self._league_slots: List[_LeagueSlot] = []
        self._league_pool: List[int] = []        # eligible update_nums sorted asc
        self._league_pool_cursor: int = 0        # for round_robin sampling
        self._league_next_refresh_slot: int = 0  # round-robin which slot to refresh
        self._league_last_refresh_t: float = 0.0
        self._league_rng = random.Random()

        if self.league_cfg is not None and getattr(self.league_cfg, "enabled", False):
            n_slots = int(self.league_cfg.n_slots)
            for _ in range(n_slots):
                m = self.cfg.make_model().to(self.device).eval()
                self._league_slots.append(_LeagueSlot(model=m))
            self._scan_league_pool()
            # Eager initial fill so workers see populated slots on step 1.
            self._fill_empty_league_slots()
            self._league_last_refresh_t = time.time()

        # Initial load
        self.resume_from_disk()

    # The Markovian env has no per-tag history to keep, so cache helpers are
    # no-ops (kept as named methods so the worker can still call them safely).
    def clear_cache(self, tag: str):
        return

    def clear_all_caches(self):
        return

    # ------------------------------------------------------------------
    # League pool management
    # ------------------------------------------------------------------

    def _scan_league_pool(self):
        """Rescan ``ckpt_dir`` and refresh the eligible league pool."""
        ckpt_dir = self.cfg.learner.ckpt_dir
        if not os.path.exists(ckpt_dir):
            self._league_pool = []
            return
        min_update = int(self.league_cfg.min_update)
        nums: List[int] = []
        for name in os.listdir(ckpt_dir):
            m = CKPT_UPDATE_PATTERN.search(name)
            if not m:
                continue
            n = int(m.group(1))
            if n >= min_update:
                nums.append(n)
        nums.sort()
        self._league_pool = nums

    def _pick_league_checkpoint(self) -> Optional[int]:
        """Pick a checkpoint update number from the pool per ``slot_sampling``."""
        pool = self._league_pool
        if not pool:
            return None
        mode = str(getattr(self.league_cfg, "slot_sampling", "uniform"))
        if mode == "round_robin":
            n = len(pool)
            upd = pool[self._league_pool_cursor % n]
            self._league_pool_cursor += 1
            return int(upd)
        if mode == "recent_weighted":
            n = len(pool)
            # Exponential weights: most recent ~e× the oldest.
            denom = max(1, n - 1)
            weights = [math.exp(i / denom) for i in range(n)]
            return int(self._league_rng.choices(pool, weights=weights, k=1)[0])
        # Default: uniform
        return int(self._league_rng.choice(pool))

    def _load_league_slot(self, slot_idx: int, update_num: int) -> bool:
        """Load ``learner_update_<update_num>.pt`` into league slot ``slot_idx``.

        Filename uses the learner's ``:06d`` zero-padded format
        (e.g. ``learner_update_000700.pt``).
        """
        path = os.path.join(
            self.cfg.learner.ckpt_dir, f"learner_update_{update_num:06d}.pt"
        )
        if not os.path.exists(path):
            logger.warning(
                f"League slot {slot_idx}: checkpoint {path} missing; skipping"
            )
            return False
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            st = _extract_state_dict(ckpt)
            clean_st = {k.replace("_orig_mod.", ""): v for k, v in st.items()}
            slot = self._league_slots[slot_idx]
            slot.model.load_state_dict(clean_st, strict=False)
            slot.update_num = int(update_num)
            logger.info(
                f"League slot {slot_idx} <- learner_update_{update_num}.pt "
                f"(pool size {len(self._league_pool)})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load league checkpoint {path}: {e}")
            return False

    def _fill_empty_league_slots(self):
        """Populate any currently-empty slots from the pool."""
        for i, slot in enumerate(self._league_slots):
            if slot.update_num is not None:
                continue
            upd = self._pick_league_checkpoint()
            if upd is None:
                return  # pool empty -- remaining slots stay empty
            self._load_league_slot(i, upd)

    def _maybe_refresh_league(self):
        """Refresh one league slot if ``refresh_every_sec`` has elapsed.

        Refresh work is amortized across ``infer_batch`` calls -- one slot
        per refresh tick, round-robin over slot indices. Pool rescan is done
        on the same tick so newly-saved checkpoints become visible.
        """
        if not self._league_slots:
            return
        now = time.time()
        if now - self._league_last_refresh_t < float(self.league_cfg.refresh_every_sec):
            return
        self._league_last_refresh_t = now
        self._scan_league_pool()
        # Fill any still-empty slots opportunistically (covers cold-start
        # when the pool was empty at __init__).
        self._fill_empty_league_slots()
        if not self._league_pool:
            return
        slot_idx = self._league_next_refresh_slot
        self._league_next_refresh_slot = (
            (slot_idx + 1) % len(self._league_slots)
        )
        upd = self._pick_league_checkpoint()
        if upd is not None:
            self._load_league_slot(slot_idx, upd)

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------

    def _resolve_policy(
        self, policy_id: int
    ) -> Tuple[nn.Module, bool, float, bool]:
        """Map ``policy_id`` -> ``(model, greedy, temp, is_current)``.

        Empty league slots fall back to the current policy so a worker that
        was assigned a slot before any checkpoints existed still gets a
        valid action.
        """
        current_greedy = bool(getattr(self, "greedy", False))
        current_temp = float(self.current_temp)
        if policy_id <= 0 or not self._league_slots:
            return self.net, current_greedy, current_temp, True
        slot_idx = int(policy_id) - 1
        if not (0 <= slot_idx < len(self._league_slots)):
            return self.net, current_greedy, current_temp, True
        slot = self._league_slots[slot_idx]
        if slot.update_num is None:
            return self.net, current_greedy, current_temp, True
        return (
            slot.model,
            bool(self.league_cfg.greedy),
            float(self.league_cfg.temp),
            False,
        )

    def _forward_group(
        self,
        model: nn.Module,
        obs_np: np.ndarray,
        greedy: bool,
        temp: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Single forward + sample for a sub-batch routed to one model.

        Returns:
            target_act       [B, P]
            frac_act         [B, P]
            logp_per_source  [B, P]  behavior logp for each source row;
                                     inactive/non-source rows are zero
            v_exp            [B]
            v_var            [B]
        """
        obs = torch.from_numpy(obs_np).to(self.device, non_blocking=True)

        # Autoregressive head: logits depend on the actions, so there is no
        # single state-only forward to sample from -- decode sequentially in
        # rank order. Mirrors the factored return contract below.
        if getattr(model, "use_ar_head", False) and getattr(model, "ar_pi_head", None) is not None:
            ar_eps = float(getattr(self, "current_behavior_eps", 0.0))
            source_np = obs_np[:, self._src_start:self._src_end]
            max_decode_steps = int((source_np > 0.5).sum(axis=1).max())
            target_act, frac_act, logp_ps, _v_logits, v_exp, v_var, _jm = model.ar_sample(
                obs, greedy=greedy, temp=temp, behavior_eps=ar_eps, max_decode_steps=max_decode_steps,
            )
            sm = obs[:, self._src_start:self._src_end].to(logp_ps.dtype)
            logp_ps = logp_ps * sm
            return (
                target_act.cpu().numpy().astype(np.int64),
                frac_act.cpu().numpy().astype(np.int64),
                logp_ps.cpu().numpy().astype(np.float32),
                v_exp.cpu().numpy().astype(np.float32),
                v_var.cpu().numpy().astype(np.float32),
            )

        pi_logits, v_logits, v_exp, joint_mask = model(obs)

        v_probs = torch.softmax(v_logits, dim=-1)
        v_support = model.v_support.to(v_probs.device, dtype=v_probs.dtype)
        v_var = (v_probs * (v_support - v_exp.unsqueeze(-1)) ** 2).sum(dim=-1)

        source_mask = obs[:, self._src_start:self._src_end].float()  # [B, P]

        B = obs.shape[0]
        P = self.max_planets
        F_dim = model.n_options
        A = P * F_dim

        pi_flat = pi_logits.reshape(B * P, A).float()
        mask_flat = joint_mask.reshape(B * P, A).float()

        top_p = float(getattr(self, "current_top_p", getattr(self, "top_p", 1.0)))

        act_flat, logp_flat, _entropy_flat = masked_sample(
            pi_flat,
            mask_flat,
            greedy=greedy,
            temp=temp,
            top_p=top_p,
        )

        act_flat = act_flat.reshape(B, P)
        logp_per_source = logp_flat.reshape(B, P)

        target_act = torch.div(act_flat, F_dim, rounding_mode="floor")
        frac_act = act_flat % F_dim

        # Keep behavior log-probability factored per source planet. PPO will
        # compute one clipped ratio per active source row, while broadcasting
        # the scalar player-perspective advantage over those active rows.
        sm = source_mask.to(logp_per_source.dtype)
        logp_per_source = logp_per_source * sm

        return (
            target_act.cpu().numpy().astype(np.int64),
            frac_act.cpu().numpy().astype(np.int64),
            logp_per_source.cpu().numpy().astype(np.float32),
            v_exp.cpu().numpy().astype(np.float32),
            v_var.cpu().numpy().astype(np.float32),
        )

    @torch.no_grad()
    def infer_batch(
        self,
        tags: List[str],
        obs_np: np.ndarray,
        policy_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Grouped batched forward + sample over the joint per-planet
        ``(target, frac)`` action.

        ``policy_ids[i] == 0``  -> current policy (``self.net``).
        ``policy_ids[i] >= 1``  -> league slot ``policy_ids[i] - 1``.

        For league rows ``logp``, ``v_exp``, ``v_var`` are returned as zeros
        (the worker drops those rows anyway). The cost is bounded by
        ``(n_distinct_policy_ids)`` forward passes per call -- in practice
        ``≤ n_slots + 1``.

        Returns five arrays in original row order:
            target_idx [B, P]   int64
            frac_idx   [B, P]   int64
            logp       [B, P]   float32, per-source behavior logp
            v_exp      [B]      float32
            v_var      [B]      float32
        """
        self._sync_weights()
        if self._league_slots:
            self._maybe_refresh_league()

        B = obs_np.shape[0]
        if len(tags) != B:
            raise ValueError(f"len(tags)={len(tags)} != batch size {B}")

        if policy_ids is None:
            policy_ids = np.zeros(B, dtype=np.int32)
        else:
            policy_ids = np.asarray(policy_ids, dtype=np.int32)
            if policy_ids.shape[0] != B:
                raise ValueError(
                    f"len(policy_ids)={policy_ids.shape[0]} != batch size {B}"
                )

        self.stats.flushes += 1
        self.stats.total_requests += B
        self.stats.avg_batch_size = (self.stats.avg_batch_size * 0.95) + (B * 0.05)

        P = self.max_planets

        out_target = np.zeros((B, P), dtype=np.int64)
        out_frac = np.zeros((B, P), dtype=np.int64)
        out_logp = np.zeros((B, P), dtype=np.float32)
        out_val = np.zeros(B, dtype=np.float32)
        out_var = np.zeros(B, dtype=np.float32)

        # Fast path: single-policy batch (league disabled or all-current step)
        # avoids the unique/loop overhead and matches pre-league behavior
        # exactly when no league rows are present.
        if not self._league_slots or np.all(policy_ids == 0):
            model, greedy, temp, _ = self._resolve_policy(0)
            t, f, lp, v, vv = self._forward_group(
                model, obs_np, greedy=greedy, temp=temp
            )
            return t, f, lp, v, vv

        # Group rows by policy_id and run one forward per group.
        for pid in np.unique(policy_ids):
            idxs = np.where(policy_ids == pid)[0]
            if idxs.size == 0:
                continue
            model, greedy, temp, is_current = self._resolve_policy(int(pid))
            sub_obs = np.ascontiguousarray(obs_np[idxs])
            t, f, lp, v, vv = self._forward_group(
                model, sub_obs, greedy=greedy, temp=temp
            )
            out_target[idxs] = t
            out_frac[idxs] = f
            # League rows: leave logp/val/var as zeros -- worker discards.
            if is_current:
                out_logp[idxs] = lp
                out_val[idxs] = v
                out_var[idxs] = vv

        return out_target, out_frac, out_logp, out_val, out_var

    # ------------------------------------------------------------------
    # DORA search support (only called when cfg.search.enabled). The normal
    # infer_batch path above is untouched, so search-off behavior is identical.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_values(self, obs_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Value-only forward over a batch of imagined next states.

        Used to score search leaves. Returns ``(v_exp [B], v_var [B])`` from the
        current policy's value head. Chunked to bound peak memory.
        """
        self._sync_weights()
        B = int(obs_np.shape[0])
        if B == 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        chunk = int(getattr(self.cfg.rollout, "infer_max_batch", 2048))
        v_out = np.zeros(B, dtype=np.float32)
        var_out = np.zeros(B, dtype=np.float32)
        for start in range(0, B, chunk):
            sub = np.ascontiguousarray(obs_np[start:start + chunk])
            obs = torch.from_numpy(sub).to(self.device, non_blocking=True)
            v_exp, v_var = self.net.forward_value(obs)
            v_out[start:start + chunk] = v_exp.cpu().numpy().astype(np.float32)
            var_out[start:start + chunk] = v_var.cpu().numpy().astype(np.float32)
        return v_out, var_out

    @torch.no_grad()
    def propose_candidates(
        self,
        obs_np: np.ndarray,
        k: int,
        temp: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample ``k`` candidate joint actions per row from the current policy.

        Returns:
            target   [B, k, P] int64
            frac     [B, k, P] int64
            logp_src [B, k, P] float32  -- per-source prior log-prob (masked to
                                          active sources; matches the per-source
                                          behavior logp the learner expects)
        """
        self._sync_weights()
        B = int(obs_np.shape[0])
        P = self.max_planets
        F_dim = self.n_options
        if B == 0:
            return (np.zeros((0, k, P), np.int64),
                    np.zeros((0, k, P), np.int64),
                    np.zeros((0, k, P), np.float32))

        obs = torch.from_numpy(np.ascontiguousarray(obs_np)).to(self.device, non_blocking=True)
        pi_logits, _v_logits, _v_exp, joint_mask = self.net(obs)
        A = P * F_dim
        pi_flat = pi_logits.reshape(B * P, A).float()
        mask_flat = joint_mask.reshape(B * P, A).float()
        source_mask = obs[:, self._src_start:self._src_end].float()  # [B, P]

        tgt_k = np.zeros((B, k, P), dtype=np.int64)
        frc_k = np.zeros((B, k, P), dtype=np.int64)
        logp_k = np.zeros((B, k, P), dtype=np.float32)
        for c in range(int(k)):
            act_flat, logp_flat, _ent = masked_sample(
                pi_flat, mask_flat, greedy=False, temp=float(temp), top_p=1.0,
            )
            act = act_flat.reshape(B, P)
            lp = logp_flat.reshape(B, P) * source_mask  # per-source prior logp
            tgt_k[:, c, :] = torch.div(act, F_dim, rounding_mode="floor").cpu().numpy()
            frc_k[:, c, :] = (act % F_dim).cpu().numpy()
            logp_k[:, c, :] = lp.cpu().numpy().astype(np.float32)
        return tgt_k, frc_k, logp_k

    def set_temp(self, temp: float):
        self.current_temp = temp

    def _sync_weights(self):
        """Pulls updated weights from the WeightStore if a new version is available."""
        # --- THE THROTTLE ---
        now = time.time()
        if self._last_sync_time:
            if now - self._last_sync_time < 5.0:
                return
        self._last_sync_time = now
        # --------------------

        try:
            # Use a lightweight version check before pulling the heavy state_dict
            latest_v = ray.get(self.weight_store.get_version.remote(), timeout=1.0)

            if latest_v > self.current_version:
                weights = ray.get(self.weight_store.get_weights.remote(), timeout=2.0)
                if weights:
                    self.net.load_state_dict(weights, strict=False)
                    self.current_version = latest_v
                    self.clear_all_caches()

                    # --- SYNC TEMPERATURE ---
                    self.current_temp = ray.get(self.weight_store.get_temp.remote())
                    # ------------------------

                    logger.info(f"InferenceActor synced to version {latest_v} | Temp: {self.current_temp:.3f}")
        except Exception:
            pass # Prevent timeout errors from crashing the GPU loop

    def resume_from_disk(self):
        """Initializes the actor with the latest weights found in the checkpoint directory."""
        if not getattr(self.cfg.learner, "resume", False):
            return

        ckpt_dir = self.cfg.learner.ckpt_dir
        if not os.path.exists(ckpt_dir):
            return

        # Find all checkpoints and load the most recent one
        import glob
        paths = sorted(glob.glob(os.path.join(ckpt_dir, "learner_update_*.pt")))

        if not paths:
            return

        latest_ckpt = paths[-1]
        try:
            st = _extract_state_dict(torch.load(latest_ckpt, map_location="cpu", weights_only=False))

            # --- THE FIX: Universal Prefix Stripping ---
            clean_st = {}
            for k, v in st.items():
                clean_st[k.replace("_orig_mod.", "")] = v

            self.net.load_state_dict(clean_st, strict=False)
            # -------------------------------------------

            logger.info(f"InferenceActor resumed from {latest_ckpt}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {latest_ckpt}: {e}")

    def refresh_snapshots_from_disk(self):
        """
        Legacy endpoint for the LearnerActor.
        In a single-policy setup, dynamic disk reloading is unnecessary because
        the latest weights are synced continuously via RAM (WeightStore).
        """
        pass

    def get_stats(self) -> dict:
        """Returns diagnostic metrics."""
        stats = {
            "total_requests": self.stats.total_requests,
            "avg_batch_size": round(self.stats.avg_batch_size, 2),
            "model_version": self.current_version,
        }
        if self._league_slots:
            stats["league_slots"] = [s.update_num for s in self._league_slots]
            stats["league_pool_size"] = len(self._league_pool)
        return stats

    def get_league_slot_bindings(self) -> List[Optional[int]]:
        """Return per-slot ``update_num`` (``None`` for empty). Used by the
        worker for optional analytics (e.g. win-rate vs checkpoint N).
        """
        return [s.update_num for s in self._league_slots]
