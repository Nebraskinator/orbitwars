"""
Learner Actor for Distributed Proximal Policy Optimization on Orbit Wars.

The LearnerActor aggregates trajectories from rollout workers, computes GAE
advantages, runs PPO updates against ``OrbitTransformer``, and broadcasts
new weights to the central WeightStore (which the InferenceActor pulls
from). Markovian env -- no per-tag history.

When the configured mode contains ``"jepa"`` we also maintain an EMA
target network over the OrbitTransformer's encoder. The target net
provides stable next-step latents for the JEPA self-prediction loss
inside ``ppo_update``; weights are blended every update with decay
``jepa_tau``.
"""

from __future__ import annotations

import asyncio
import copy
import glob
import logging
import os
import random
import traceback
from typing import Dict, List, Optional, Tuple, Final

import ray
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import RunConfig
import ppo_core
from ppo_core import AsyncEpisodeDataset, gae_from_episode, ppo_update


logger = logging.getLogger(__name__)


class LearnerActor:
    """
    Ray actor that owns model optimization and weight distribution.

    Supported training modes (mirrors ``ppo_core.ppo_update``):
        "ppo"        -- factored (target, frac) PPO with clipped ratio
        "imitation"  -- behavioral cloning on per-source factored actions
        "warmup"     -- critic-only update (frozen actor)
    """

    def __init__(self, cfg: RunConfig, weight_store: ray.actor.ActorHandle):
        self.run_cfg = cfg
        self.cfg = cfg.learner
        self.weight_store = weight_store

        if self.cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("LearnerActor configured for CUDA, but no GPU was detected.")

        self.net: Optional[nn.Module] = None
        self.opt: Optional[optim.Optimizer] = None
        self.sched: Optional[optim.lr_scheduler.LambdaLR] = None

        # EMA target net for JEPA. Built lazily in ``_init_if_needed`` when
        # ``self.cfg.mode`` contains "jepa"; ``None`` for plain PPO.
        self.target_net: Optional[nn.Module] = None
        self.jepa_tau: float = 0.99

        # CPU-side experience buffer; moved to GPU only inside PPO minibatches.
        # ``act_dim`` is a no-op field in the dataset (kept for API parity);
        # 0 is fine because the per-planet (target, frac) shape is never
        # referenced through it.
        self.dataset = AsyncEpisodeDataset(act_dim=0, device="cpu")

        self.update_idx = 0
        self.total_episodes = 0
        self.total_steps = 0

        self._q: asyncio.Queue = asyncio.Queue(
            maxsize=int(self.run_cfg.rollout.learn_max_pending_batches)
        )

        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)

        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._loop())
        self._task.add_done_callback(self._on_loop_done)
        self._startup_task = loop.create_task(self._maybe_resume_latest())

        self.ep_counter = 0

    # ------------------------------------------------------------------
    # Optimizer / scheduler / mode wiring
    # ------------------------------------------------------------------

    def _init_optimizer(self):
        if self.net is None:
            return

        zones = self.net.get_parameter_zones()
        active_grads, lr_mults = self._get_zone_config()

        # Apply gradient toggles.
        for zone_name, params in zones.items():
            requires_grad = active_grads.get(zone_name, True)
            for _, p in params:
                p.requires_grad = requires_grad

        # Build parameter groups with weight-decay logic.
        wd_val = float(getattr(self.cfg, "weight_decay", 0.01))
        base_lr = float(self.cfg.lr)
        param_groups = []

        for zone_name, params in zones.items():
            if not active_grads.get(zone_name, True):
                continue
            decay, no_decay = [], []
            for name, p in params:
                # No decay on LayerNorm, biases, embeddings, or 1D tensors.
                no_decay_condition = (
                    any(x in name for x in ["_emb", "norm"])
                    or name.endswith(".bias")
                    or p.ndim <= 1
                    or name.endswith("attn_scale")
                    or name.endswith("pair_scale")
                )
                (no_decay if no_decay_condition else decay).append(p)

            target_lr = base_lr * lr_mults.get(zone_name, 1.0)
            if decay:
                param_groups.append({"params": decay, "lr": target_lr,
                                     "weight_decay": wd_val,
                                     "name": f"{zone_name}_wd"})
            if no_decay:
                param_groups.append({"params": no_decay, "lr": target_lr,
                                     "weight_decay": 0.0,
                                     "name": f"{zone_name}_stable"})

        self.opt = optim.AdamW(param_groups, eps=1e-5)
        self._init_scheduler()
    
    def _debug_scale_optimizer_membership(self):
        base_net = self.net._orig_mod if hasattr(self.net, "_orig_mod") else self.net
    
        opt_ids = {}
        for group in self.opt.param_groups:
            for p in group["params"]:
                opt_ids[id(p)] = group
    
        print("\n[scale optimizer membership]", flush=True)
    
        for name, p in base_net.named_parameters():
            if (
                "node_pair_trunk" in name
                and (
                    name.endswith("attn_scale")
                    or name.endswith("pair_scale")
                )
            ):
                group = opt_ids.get(id(p))
                print(
                    f"{name}: "
                    f"dtype={p.dtype}, "
                    f"requires_grad={p.requires_grad}, "
                    f"in_optimizer={group is not None}, "
                    f"group={group.get('name') if group else None}, "
                    f"lr={group.get('lr') if group else None}, "
                    f"wd={group.get('weight_decay') if group else None}",
                    flush=True,
                )
                
    def _get_zone_config(self) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """Mode-aware gradient toggles and per-zone LR multipliers.

        Zones come from ``OrbitTransformer.get_parameter_zones``: subnets,
        transformer, readout, jepa, pi, v, embeddings.
        """
        mode = self.cfg.mode

        grad_toggles = {
            "imitation":                 {"embeddings": True,  "subnets": True,  "transformer": True,  "jepa": False, "readout": True,  "pi": True,  "v": True},
            "imitation_with_jepa":       {"embeddings": True,  "subnets": True,  "transformer": True,  "jepa": True,  "readout": True,  "pi": True,  "v": True},
            "imitation_with_td_jepa":    {"embeddings": True,  "subnets": True,  "transformer": True,  "jepa": True,  "readout": True,  "pi": True,  "v": True},
            "imitation_frozen_backbone": {"embeddings": False, "subnets": False, "transformer": False, "jepa": False, "readout": True,  "pi": True,  "v": True},
            "jepa_pretraining":          {"embeddings": True,  "subnets": True,  "transformer": True,  "jepa": True,  "readout": False, "pi": False, "v": False},
            "warmup":                    {"embeddings": False, "subnets": False, "transformer": False, "jepa": False, "readout": True, "pi": False, "v": True},
            "warmup_with_actor_reset":   {"embeddings": False, "subnets": False, "transformer": False, "jepa": False, "readout": True, "pi": True,  "v": True},
            "ppo_frozen_backbone":       {"embeddings": False, "subnets": False, "transformer": False, "jepa": False, "readout": True,  "pi": True,  "v": True},
            "ppo":                       {"embeddings": True,  "subnets": True,  "transformer": True,  "jepa": False, "readout": True,  "pi": True,  "v": True},
            "ppo_with_jepa":             {"embeddings": True,  "subnets": True,  "transformer": True,  "jepa": True,  "readout": True,  "pi": True,  "v": True},
            "ppo_with_td_jepa":          {"embeddings": True,  "subnets": True,  "transformer": True,  "jepa": True,  "readout": True,  "pi": True,  "v": True},
        }

        lr_mult_configs = {
            "imitation":                 {"embeddings": 1.0, "subnets": 1.0, "transformer": 1.0, "jepa": 0.0, "readout": 1.0, "pi": 1.0, "v": 1.0},
            "imitation_frozen_backbone": {"embeddings": 0.0, "subnets": 0.0, "transformer": 0.0, "jepa": 0.0, "readout": 1.0, "pi": 1.0, "v": 2.0},
            "imitation_with_jepa":       {"embeddings": 1.0, "subnets": 1.0, "transformer": 1.0, "jepa": 1.0, "readout": 1.0, "pi": 1.0, "v": 1.0},
            "imitation_with_td_jepa":    {"embeddings": 1.0, "subnets": 1.0, "transformer": 1.0, "jepa": 1.0, "readout": 1.0, "pi": 1.0, "v": 1.0},
            "jepa_pretraining":          {"embeddings": 1.0, "subnets": 1.0, "transformer": 1.0, "jepa": 1.0, "readout": 0.0, "pi": 0.0, "v": 0.0},
            "warmup":                    {"embeddings": 0.0, "subnets": 0.0, "transformer": 0.0, "jepa": 0.0, "readout": 1.0, "pi": 0.0, "v": 1.0},
            "warmup_with_actor_reset":   {"embeddings": 0.0, "subnets": 0.0, "transformer": 0.0, "jepa": 0.0, "readout": 1.0, "pi": 1.0, "v": 1.0},
            "ppo_frozen_backbone":       {"embeddings": 0.0, "subnets": 0.0, "transformer": 0.0, "jepa": 0.0, "readout": 1.0, "pi": 1.0, "v": 2.0},
            "ppo_with_jepa":             {"embeddings": 0.5, "subnets": 0.5, "transformer": 0.5, "jepa": 1.0, "readout": 1.0, "pi": 1.0, "v": 2.0},
            "ppo_with_td_jepa":          {"embeddings": 0.5, "subnets": 0.5, "transformer": 0.5, "jepa": 1.0, "readout": 1.0, "pi": 1.0, "v": 2.0},
            "ppo":                       {"embeddings": 0.5, "subnets": 0.5, "transformer": 0.5, "jepa": 0.0, "readout": 1.0, "pi": 1.0, "v": 1.0},
        }

        if mode not in grad_toggles or mode not in lr_mult_configs:
            raise ValueError(f"Unknown training mode: {mode!r}. "
                             f"Supported: {list(grad_toggles.keys())}")
        return grad_toggles[mode], lr_mult_configs[mode]

    def _init_scheduler(self):
        w_steps = int(getattr(self.cfg, "lr_warmup_steps", 0))
        h_steps = int(getattr(self.cfg, "lr_hold_steps", 0))
        t_steps = int(getattr(self.cfg, "lr_total_steps", 0))

        def lr_lambda(step: int) -> float:
            if step < w_steps:
                return float(step + 1) / float(max(w_steps, 1))
            if step < (w_steps + h_steps):
                return 1.0
            anneal_start = w_steps + h_steps
            progress = min(1.0, (step - anneal_start) / max(1, t_steps - anneal_start))
            return 1.0 / ((8 * progress + 1) ** 1.5)

        self.sched = optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lr_lambda)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    async def _loop(self):
        self._init_if_needed()

        while True:
            msgs = [await self._q.get()]
            while not self._q.empty():
                try:
                    msgs.append(self._q.get_nowait())
                except asyncio.QueueEmpty:
                    break

            for msg in msgs:
                if not isinstance(msg, tuple) or msg[0] != "packed":
                    continue
                (_, obs_cat, act_cat, logp_cat, val_cat, var_cat, rew_cat,
                 done_cat, lengths, sbr_cat, searched_cat) = msg

                obs_all = torch.from_numpy(obs_cat)
                # ``act`` is now per-planet [T, P, 2]; keep int64.
                act_all = torch.from_numpy(act_cat).long()
                val_all = torch.from_numpy(val_cat).float()
                var_all = torch.from_numpy(var_cat).float()
                rew_all = torch.from_numpy(rew_cat).float()
                done_all = torch.from_numpy(done_cat).float()

                # Per-episode GAE.
                adv_chunks, ret_chunks, id_chunks = [], [], []
                curr = 0
                for length in lengths.tolist():
                    end = curr + int(length)
                    adv, ret = gae_from_episode(
                        rewards=rew_all[curr:end],
                        values=val_all[curr:end],
                        dones=done_all[curr:end],
                        gamma=self.cfg.gamma,
                        lam=self.cfg.gae_lambda,
                        variances=var_all[curr:end] if self.cfg.use_dynamic_lambda else None,
                        lam_min=self.cfg.dynamic_lambda_min,
                        lam_max=self.cfg.dynamic_lambda_max,
                        lam_eps=self.cfg.dynamic_lambda_eps,
                    )
                    adv_chunks.append(adv)
                    ret_chunks.append(ret)
                    id_chunks.append(torch.full((int(length),),
                                                self.ep_counter, dtype=torch.long))
                    curr = end
                    self.ep_counter += 1

                self.dataset.add_steps(
                    obs_all, act_all, torch.from_numpy(logp_cat), val_all,
                    torch.cat(adv_chunks), torch.cat(ret_chunks),
                    # Unused dataset slot; kept zero-filled for API parity.
                    torch.zeros(len(act_all)),
                    torch.cat(id_chunks),
                )

                self.total_episodes += len(lengths)
                self.total_steps += len(act_all)

            if len(self.dataset) >= self.cfg.steps_per_update:
                await self._perform_update()

    @staticmethod
    def _stats_1d(x: torch.Tensor, prefix: str) -> Dict[str, float]:
        if x is None:
            return {f"{prefix}_n": 0.0}
        x = x.detach()
        if x.numel() == 0:
            return {f"{prefix}_n": 0.0}
        x = x.float().view(-1)

        finite = torch.isfinite(x)
        xf = x[finite] if not bool(finite.all()) else x
        if xf.numel() == 0:
            return {f"{prefix}_n": float(x.numel()), f"{prefix}_finite": 0.0}

        def q(p: float) -> float:
            return float(torch.quantile(xf, torch.tensor(p, device=xf.device)))

        return {
            f"{prefix}_n": float(x.numel()),
            f"{prefix}_finite": float(xf.numel()),
            f"{prefix}_mean": float(xf.mean()),
            f"{prefix}_std":  float(xf.std(unbiased=False)),
            f"{prefix}_min":  float(xf.min()),
            f"{prefix}_p05":  q(0.05),
            f"{prefix}_p50":  q(0.50),
            f"{prefix}_p95":  q(0.95),
            f"{prefix}_max":  float(xf.max()),
            f"{prefix}_abs_mean": float(xf.abs().mean()),
            f"{prefix}_abs_max":  float(xf.abs().max()),
        }

    async def _perform_update(self):
        """Runs one PPO update and broadcasts the new weights."""
        try:
            (obs_u, act_u, logp_u, val_u, adv_u, ret_u,
             next_hp_u, ep_ids_u) = self.dataset.swap_out_tensor_cache()

            diag = {}
            diag.update(self._stats_1d(adv_u, "adv_pre"))
            diag.update(self._stats_1d(ret_u, "ret"))
            diag.update(self._stats_1d(val_u, "v_old"))
            diag.update(self._stats_1d(logp_u, "logp_old"))

            # Advantage normalization.
            adv_u = (adv_u - adv_u.mean()) / (adv_u.std() + 1e-8)
            adv_u = torch.clamp(adv_u, -10.0, 10.0)
            diag["adv_norm_mean"] = float(adv_u.mean())
            diag["adv_norm_std"]  = float(adv_u.std(unbiased=False))

            train_ds = AsyncEpisodeDataset(act_dim=0, device=str(obs_u.device))
            train_ds.add_steps(obs_u, act_u, logp_u, val_u,
                               adv_u, ret_u, next_hp_u, ep_ids_u)

            ppo_kwargs = self.cfg.ppo_kwargs()
            stats = ppo_update(
                net=self.net, opt=self.opt, dataset=train_ds, scheduler=self.sched,
                mode=self.cfg.mode, target_net=self.target_net,
                **ppo_kwargs,
            )

            # After the online net moves, drift the EMA target a step toward it
            # so the next update sees a slightly fresher target.
            if self.target_net is not None:
                self._update_target_net()

            self.update_idx += 1

            # Broadcast weights.
            base_net = self.net._orig_mod if hasattr(self.net, "_orig_mod") else self.net
            weights = {k: v.cpu().detach() for k, v in base_net.state_dict().items()}
            self.weight_store.update.remote(weights, self.update_idx)
            new_temp = self.cfg.get_temp(self.total_steps)
            self.weight_store.set_temp.remote(new_temp)

            logger.info(
                f"Update {self.update_idx}: Loss={stats.total_loss:.3f}, "
                f"KL={stats.approx_kl:.4f}"
            )
            print(
                f"[learner] upd={self.update_idx} "
                f"kl={stats.approx_kl:.4f} clip={stats.clip_frac:.3f} "
                f"ent/src={stats.entropy:.3f} "
                f"vloss={stats.v_loss:.3f} "
                f"ploss={stats.pg_loss:.3f} "
                f"jepa={stats.jepa_loss:.4f} "
                f"loss={stats.total_loss:.3f} "
                f"n_mb={stats.n_mb}"
            )

            if self.update_idx % self.cfg.save_every_updates == 0:
                self._save_checkpoint(self._ckpt_path_for_update(self.update_idx))

        except Exception as e:
            logger.error(f"PPO Update failed: {e}")
            traceback.print_exc()
            self.dataset.clear()

    # ------------------------------------------------------------------
    # I/O endpoints (called by RolloutWorker / SyncLearnerClient)
    # ------------------------------------------------------------------

    async def submit_packed_batch(
        self,
        obs_cat: np.ndarray,
        act_cat: np.ndarray,    # [S, P, 2]
        logp_cat: np.ndarray,   # [S, P] per-source behavior logp
        val_cat: np.ndarray,    # [S]
        var_cat: np.ndarray,    # [S]
        rew_cat: np.ndarray,    # [S]
        done_cat: np.ndarray,   # [S]
        lengths: np.ndarray,    # [B]
        sbr_cat: Optional[np.ndarray] = None,       # [S, P, 2] SBR joint action
        searched_cat: Optional[np.ndarray] = None,  # [S] searched flag
    ):
        await self._q.put(("packed", obs_cat, act_cat, logp_cat, val_cat,
                           var_cat, rew_cat, done_cat, lengths,
                           sbr_cat, searched_cat))
        return True

    async def save_now(self, path: Optional[str] = None) -> str:
        if self.net is None or self.opt is None:
            raise RuntimeError("Cannot save: model not initialized yet.")
        if path is None:
            path = self._ckpt_path_for_update(self.update_idx)
        self._save_checkpoint(path)
        return path

    async def get_stats(self) -> dict:
        return {
            "update": self.update_idx,
            "episodes": self.total_episodes,
            "steps_in_dataset": len(self.dataset),
            "total_steps": self.total_steps,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _ckpt_path_for_update(self, update_idx: int) -> str:
        return os.path.join(self.cfg.ckpt_dir, f"learner_update_{update_idx:06d}.pt")

    def _save_checkpoint(self, path: str):
        base_net = self.net._orig_mod if hasattr(self.net, "_orig_mod") else self.net
        payload = {
            "model": base_net.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict() if self.sched else None,
            "update_idx": self.update_idx,
            "total_steps": self.total_steps,
            "run_cfg": self.run_cfg.as_dict(),
            "total_episodes": self.total_episodes,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        }
        torch.save(payload, path + ".tmp")
        os.replace(path + ".tmp", path)

    def _init_if_needed(self):
        if self.net is None:
            # 1. Instantiate the primary online network
            self.net = self.run_cfg.make_model().to(self.cfg.device).train()
            self.net.enable_bf16_recurrent_path()
            
            # 2. Build the JEPA target network by instantiating a clean secondary module
            if "jepa" in self.cfg.mode:
                # Instantiate freshly from factory to avoid deepcopy graph cloning bugs
                self.target_net = self.run_cfg.make_model().to(self.cfg.device).eval()
                #self.target_net.enable_bf16_recurrent_path()
                
                # Copy the initial weights directly via state dict
                self.target_net.load_state_dict(self.net.state_dict())
                
                for p in self.target_net.parameters():
                    p.requires_grad = False
                    
                logger.info("LearnerActor built JEPA EMA target net "
                            f"(tau={self.jepa_tau})")
                
                # Compile the target net cleanly
                self.target_net = torch.compile(self.target_net, mode="reduce-overhead")
            
            # 3. Compile the main online network
            self.net = torch.compile(self.net, mode="reduce-overhead")
            
            # 4. Initialize the parameter groups and optimizer
            self._init_optimizer()
            #self._debug_scale_optimizer_membership()

    @torch.no_grad()
    def _update_target_net(self, tau: Optional[float] = None) -> None:
        """In-place EMA: target = tau * target + (1 - tau) * online."""
        if self.target_net is None or self.net is None:
            return
        t = float(self.jepa_tau if tau is None else tau)
        for p_t, p_o in zip(self.target_net.parameters(),
                            self.net.parameters()):
            p_t.data.mul_(t).add_(p_o.data, alpha=1.0 - t)
        # Buffers (positional embeddings, type_ids, v_support) are tied to
        # schema, not learned -- copy verbatim so any future buffer additions
        # stay in sync without code changes.
        for b_t, b_o in zip(self.target_net.buffers(), self.net.buffers()):
            b_t.data.copy_(b_o.data)

    def _latest_ckpt_path(self) -> Optional[str]:
        paths = sorted(glob.glob(
            os.path.join(self.cfg.ckpt_dir, "learner_update_*.pt")
        ))
        return paths[-1] if paths else None

    async def _maybe_resume_latest(self):
        self._init_if_needed()
        assert self.net is not None

        loaded = False
        if self.cfg.resume:
            path = self._latest_ckpt_path()
            if path:
                try:
                    self._load_checkpoint(path)
                    loaded = True
                    print(f"[learner] resumed from {path}", flush=True)
                except Exception as e:
                    print(f"[learner] resume failed from {path}: {e!r}", flush=True)

        # Publish weights at startup either way.
        base_net = self.net._orig_mod if hasattr(self.net, "_orig_mod") else self.net
        sd_cpu = {k: v.detach().to("cpu") for k, v in base_net.state_dict().items()}
        self.weight_store.update.remote(sd_cpu, self.update_idx)
        print(
            f"[learner] pushed {'resumed' if loaded else 'init'} policy 0 to inference",
            flush=True,
        )

    def _on_loop_done(self, task: asyncio.Task):
        if not task.cancelled() and task.exception():
            logger.critical("Learner training loop CRASHED!")
            traceback.print_exception(None, task.exception(), None)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self._init_if_needed()
        assert self.net is not None and self.opt is not None

        ckpt_mode = ckpt.get("run_cfg", {}).get("learner", {}).get("mode", "unknown")
        current_mode = self.cfg.mode
        should_reset_opt = (current_mode != ckpt_mode)

        # --- THE FIX: Universal Prefix Stripping ---
        # 1. Strip the prefix from the checkpoint's keys
        ckpt_weights_clean = {}
        for k, v in ckpt["model"].items():
            ckpt_weights_clean[k.replace("_orig_mod.", "")] = v

        # 2. Get the underlying uncompiled network to load into
        base_net = self.net._orig_mod if hasattr(self.net, "_orig_mod") else self.net
        current_state = base_net.state_dict()
        
        model_weights = {}
        for k, v in ckpt_weights_clean.items():
            if "attn_mask" in k:
                continue

            # v_support is a runtime-config buffer derived from v_min/v_max/v_bins.
            # Do not load it from the checkpoint, otherwise changing v_min/v_max
            # at runtime has no effect when v_bins is unchanged.
            if k == "v_support":
                print("[learner] Keeping runtime v_support; skipped checkpoint v_support.")
                continue
                
            # Drop weights if the shapes no longer match (e.g. changing v_bins)
            if k in current_state and current_state[k].shape != v.shape:
                print(f"[learner] Dropping {k} due to shape mismatch "
                      f"(ckpt: {list(v.shape)}, model: {list(current_state[k].shape)})")
                continue
                
            model_weights[k] = v
            
        # Load safely into the base network
        base_net.load_state_dict(model_weights, strict=False)

        # Re-seed the JEPA target net from the freshly-loaded online weights.
        if self.target_net is not None:
            base_target = self.target_net._orig_mod if hasattr(self.target_net, "_orig_mod") else self.target_net
            base_target.load_state_dict(base_net.state_dict(), strict=False)
            
        if should_reset_opt:
            print(f"[learner] PHASE CHANGE ({ckpt_mode} -> {current_mode}). "
                  "Skipping optimizer/scheduler load to enforce fresh LR.")
            
            if current_mode == "warmup_with_actor_reset":
                print("[learner] Resetting actor head parameters for warmup phase.")
                pi_head = getattr(self.net, "pi_head", None)
                if pi_head is not None and hasattr(pi_head, "reset_parameters"):
                    # EdgeActionHead.reset_parameters re-inits each
                    # sub-Linear with std=0.02 and zeroes the final
                    # target/option projections so the post-mask logits
                    # are flat at the start of the warmup phase.
                    pi_head.reset_parameters()
                elif pi_head is not None and hasattr(pi_head, "weight"):
                    # Back-compat: a plain nn.Linear pi_head.
                    nn.init.normal_(pi_head.weight, std=0.02)
                    if getattr(pi_head, "bias", None) is not None:
                        nn.init.zeros_(pi_head.bias)
                        
            self.update_idx = 0
            self.total_episodes = 0
            self.total_steps = 0
            return

        # Same-phase resume.
        try:
            self.opt.load_state_dict(ckpt["optimizer"])
            print("[learner] Optimizer state loaded successfully.")
        except Exception:
            print("[learner] Optimizer topology mismatch. Resetting optimizer state.")

        if self.sched is not None and "scheduler" in ckpt:
            try:
                self.sched.load_state_dict(ckpt["scheduler"])
                print("[learner] Scheduler state loaded successfully.")
            except Exception:
                pass

        # Apply fresh LR / WD baselines.
        try:
            new_base_lr = float(self.cfg.lr)
            active_grads, lr_mults = self._get_zone_config()

            for pg in self.opt.param_groups:
                group_name = pg.get("name", "")
                zone = group_name.split("_")[0]
                if zone in lr_mults and active_grads.get(zone, False):
                    target_lr = new_base_lr * lr_mults[zone]
                    pg["lr"] = target_lr
                    pg["initial_lr"] = target_lr

            if self.sched is not None:
                self.sched.base_lrs = [pg["initial_lr"] for pg in self.opt.param_groups]

            new_wd = float(getattr(self.cfg, "weight_decay", 0.01))
            for pg in self.opt.param_groups:
                if pg["weight_decay"] > 0:
                    pg["weight_decay"] = new_wd

            print(f"[learner] LR jump applied dynamically. base_lr={new_base_lr:.2g}")
        except Exception as e:
            print(f"[learner] LR jump override failed: {e!r}")

        self.update_idx = int(ckpt.get("update_idx", 0))
        self.total_episodes = int(ckpt.get("total_episodes", 0))
        self.total_steps = int(ckpt.get("total_steps", 0))

        new_temp = self.cfg.get_temp(self.total_steps)
        self.weight_store.set_temp.remote(new_temp)

        if "torch_rng" in ckpt:
            rng = ckpt["torch_rng"]
            if isinstance(rng, torch.Tensor):
                rng = rng.detach().to("cpu")
                if rng.dtype != torch.uint8:
                    rng = rng.to(torch.uint8)
            torch.set_rng_state(rng)
        if "numpy_rng" in ckpt:
            np.random.set_state(ckpt["numpy_rng"])
        if "python_rng" in ckpt:
            random.setstate(ckpt["python_rng"])
