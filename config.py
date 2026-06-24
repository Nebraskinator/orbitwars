"""
Configuration for Orbit Wars reinforcement learning.

Schema-derived sizes (token dimensions, MAX_PLANETS, action buckets) are NOT
mirrored here -- they live in ``orbit_obs_reftrace.schema_metadata()`` and flow
through ``RunConfig.make_model``.

The ``ModelConfig`` defaults below describe the exact architecture of the
released submission. Training and inference both build the full model
(``make_model`` does not pass ``inference_only``), so a checkpoint produced by
this pipeline loads back through ``main.py`` for inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    """Architectural knobs for ``OrbitTransformer``.

    Input dimensions and head sizes are read from
    ``schema_metadata()`` at construction time; this config only carries the
    choices the schema does not dictate. Values are the released submission's.
    """
    d_model: int = 144
    pair_dim: int = 160
    # Fixed hidden width of the per-layer pair transition FF. Not scaled with
    # pair_dim. (Accepted by OrbitTransformer for forward-compat; the released
    # trunk sizes its pair FF internally.)
    pair_hidden_dim: int = 128
    # Whether contextualized token information is written back into the pair
    # stream each trunk layer. Always constructed so checkpoints keep an
    # identical state_dict schema; this flag only gates its use in forward.
    pair_token_writeback: bool = True
    # Recompute trunk blocks in backward instead of saving intermediates.
    # Training-only; trades trunk forward FLOPs for lower activation memory.
    grad_checkpoint: bool = True
    token_bottleneck_div: float = 2.0
    pair_bottleneck_div: float = 4.0
    n_layers: int = 5
    n_heads: int = 4
    ff_expansion: float = 2.0
    dropout: float = 0.0
    use_type_embedding: bool = True
    use_triangles: bool = False
    inference_only: bool = False

    # Autoregressive (coordinated) action head. The released submission used the
    # factored per-source policy head, so this stays off.
    use_ar_head: bool = False
    ar_head_layers: int = 1
    ar_head_n_heads: int = 4


@dataclass(frozen=True)
class EnvConfig:
    """Orbit Wars environment settings.

    ``n_agents`` is a tuple of allowed per-match player counts. ``(2,)`` is
    1v1 self-play, ``(4,)`` is FFA; with multiple values each rollout worker
    balances its matches across the choices by player-slot count.
    """
    env_name: str = "orbit_wars"
    max_steps: int = 500              # episode length cap
    n_agents: Tuple[int, ...] = (2, 4)


@dataclass(frozen=True)
class RolloutConfig:
    """Settings for distributed rollout workers and batching logic."""
    target_concurrent_battles: int = 384
    rooms_per_pair: int = 32

    # Timeouts (seconds)
    infer_timeout_s: float = 15.0
    open_timeout: float = 30.0

    # Inference batching constraints
    infer_min_batch: int = 32
    infer_max_batch: int = 2048
    infer_wait_ms: float = 3.0
    infer_max_pending: int = 20000

    # Learner submission limits
    learn_min_episodes: int = 1
    learn_max_episodes: int = 8
    learn_wait_ms: float = 1.0
    learn_max_pending_episodes: int = 3
    learn_max_pending_batches: int = 1

    def worker_kwargs(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferenceConfig:
    """Config for the centralized inference actor."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    server_min_batch: int = 0
    server_max_wait: float = 0

    def kwargs(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "server_min_batch": self.server_min_batch,
            "server_max_wait": self.server_max_wait,
        }


@dataclass(frozen=True)
class AugmentConfig:
    """Inline GPU minibatch observation augmentation (see ``obs_augment.py``).

    Applied to each PPO minibatch after it lands on the training device. Both
    transforms are label-preserving (planet ordering, masks, and the
    ``(target_idx, option_idx)`` action are untouched), so no downstream
    remapping is needed.
    """
    enabled: bool = False
    # Random C4 rotation (90-degree quarter-turns) about the sun center.
    rotate: bool = True
    # Random permutation of the enemy owner slots {1, 2, 3}.
    permute_players: bool = False

    def kwargs(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LearnerConfig:
    """Core PPO and hyperparameter settings."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Supported modes (routed inside ppo_core.ppo_update):
    #   "ppo"        -- factored (target, frac) PPO with clipped ratio
    #   "imitation"  -- behavioral cloning on per-source factored actions
    #   "warmup"     -- critic-only update (frozen actor)
    mode: str = "ppo"

    # GAE
    gamma: float = 0.999
    gae_lambda: float = 0.97
    use_dynamic_lambda: bool = False
    dynamic_lambda_min: float = 0.55
    dynamic_lambda_max: float = 0.95
    dynamic_lambda_eps: float = 1e-8

    # Distributional value head. Bin centers are uniformly spaced in symlog
    # space; targets use Gaussian-smoothed HL-Gauss labels (sigma in bin units).
    # The released value head has 101 bins over [-5, 5].
    use_twohot_value: bool = False
    v_min: float = -5.0
    v_max: float = 5.0
    v_bins: int = 101
    value_hl_sigma: float = 0.75

    # Schedules
    temp_start: float = 1.0
    temp_end: float = 1.0
    temp_total_steps: int = 500_000
    explore_eps: float = 0.0
    bc_label_smoothing: float = 0.01

    # Optimizer
    lr: float = 3e-4
    lr_warmup_steps: int = 1_000
    lr_hold_steps: int = 500_000
    lr_total_steps: int = 1_500_000
    weight_decay: float = 0.0

    # Layer-specific LR multipliers (filled in __post_init__)
    lr_backbone_mult: float = field(init=False)
    lr_pi_mult: float = field(init=False)
    lr_v_mult: float = field(init=False)

    # PPO
    update_epochs: int = 4
    minibatch_size: int = 256
    grad_accum_steps: int = 2
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    clip_vloss: bool = False
    max_grad_norm: float = 10
    target_kl: Optional[float] = 0.02
    steps_per_update: int = 9216

    # PPO source-balancing mode (see ppo_core.ppo_update):
    #   "source_mean"     : per-source clipped ratios, mean over active sources
    #   "source_sqrt_sum" : divide source loss sums by sqrt(active sources)
    #   "joint_sum"       : summed-joint PPO; one clipped ratio per state
    ppo_source_balance: str = "joint_sum"

    # Inline GPU minibatch observation augmentation.
    augment: AugmentConfig = field(default_factory=AugmentConfig)

    # Checkpointing
    ckpt_dir: str = "checkpoints"
    save_every_updates: int = 25
    keep_last: int = 500
    resume: bool = True

    def __post_init__(self):
        """Sets per-group gradient multipliers from the training mode."""
        multipliers = {
            "imitation": (1.0, 1.0, 0.6),  # backbone, actor, critic
            "warmup":    (0.0, 0.0, 1.0),
            "ppo":       (0.5, 1.0, 2.0),
        }
        backbone, pi, v = multipliers.get(self.mode, (1.0, 1.0, 1.0))
        # object.__setattr__ because the dataclass is frozen
        object.__setattr__(self, "lr_backbone_mult", backbone)
        object.__setattr__(self, "lr_pi_mult", pi)
        object.__setattr__(self, "lr_v_mult", v)

    def get_temp(self, global_step: int) -> float:
        """Linear annealing of the action temperature."""
        if global_step >= self.temp_total_steps:
            return self.temp_end
        frac = global_step / self.temp_total_steps
        return self.temp_start + frac * (self.temp_end - self.temp_start)

    def ppo_kwargs(self) -> Dict[str, Any]:
        """Hyperparameters consumed by ``ppo_core.ppo_update``."""
        keys = [
            "update_epochs", "minibatch_size", "grad_accum_steps",
            "clip_coef", "ent_coef", "vf_coef", "clip_vloss",
            "max_grad_norm", "target_kl", "bc_label_smoothing",
            "v_min", "v_max", "v_bins", "value_hl_sigma", "ppo_source_balance",
            "gamma", "explore_eps",
        ]
        out = {k: getattr(self, k) for k in keys}
        out["augment"] = self.augment.kwargs()
        return out


@dataclass(frozen=True)
class RewardConfig:
    """Reward signal. Two summed components:

    * Terminal: ``terminal_win`` / ``terminal_loss`` on the last step.
    * Dense production shaping: each step, each agent receives
      ``production_coef * (prod_total_t - prod_total_{t-1})`` over owned planets.
    """
    terminal_win: float = 1.
    terminal_loss: float = -1.
    production_coef: float = 1.0


@dataclass(frozen=True)
class LeagueConfig:
    """League play: a subset of agent slots play older checkpoints instead of
    the current policy. League agents are opponents only -- their trajectories
    are not submitted to the learner.
    """
    enabled: bool = False
    # Fraction of agent slots (per worker) bound to league policies.
    fraction: float = 0.25
    # Resident checkpoint slots on the InferenceActor; caps per-step forwards.
    n_slots: int = 4
    # Skip checkpoints below this learner-update index (near-random init).
    min_update: int = 125
    # Cadence at which a fresh checkpoint is swapped into a slot.
    refresh_every_sec: float = 60.0
    # Slot refill policy: "uniform" | "recent_weighted" | "round_robin".
    slot_sampling: str = "uniform"
    # League action sampling.
    greedy: bool = True
    temp: float = 1.0


@dataclass(frozen=True)
class SearchConfig:
    """1-ply sampled best-response (SBR) search. ``enabled=False`` is a hard
    no-op and is the released configuration; the self-play search subtree is
    not included in this reproduction repo.
    """
    enabled: bool = False
    n_self_samples: int = 6
    n_opp_samples: int = 3
    depth: int = 1
    gamma: float = 0.997
    aggregation: str = "mean"
    max_eval_batch: int = 2048
    oversample: int = 4
    proposal_mode: str = "sample"
    gate_mode: str = "all"
    gate_fraction: float = 0.25
    distill_coef: float = 1.0
    actor_num_gpus: float = 0.0
    backend: str = "numpy"
    numpy_parallel: bool = True

    def kwargs(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunConfig:
    """Master configuration object coordinating all sub-configs."""
    model: ModelConfig
    env: EnvConfig
    rollout: RolloutConfig
    infer: InferenceConfig
    learner: LearnerConfig
    reward: RewardConfig
    league: LeagueConfig
    search: SearchConfig = field(default_factory=SearchConfig)

    @classmethod
    def default(cls) -> "RunConfig":
        return cls(
            model=ModelConfig(),
            env=EnvConfig(),
            rollout=RolloutConfig(),
            infer=InferenceConfig(),
            learner=LearnerConfig(),
            reward=RewardConfig(),
            league=LeagueConfig(),
            search=SearchConfig(),
        )

    def make_model(self) -> nn.Module:
        """Instantiates the OrbitTransformer from the live observation schema.

        Schema-derived sizes (token dims, MAX_PLANETS, action buckets) come
        from ``schema_metadata()``.
        """
        try:
            from orbit_obs_reftrace import schema_metadata
            from ppo_core import OrbitTransformer
        except ImportError as e:
            logger.error(f"Failed to import model components: {e}")
            raise

        meta = schema_metadata()
        return OrbitTransformer(
            meta=meta,
            d_model=self.model.d_model,
            pair_dim=self.model.pair_dim,
            pair_hidden_dim=self.model.pair_hidden_dim,
            pair_token_writeback=self.model.pair_token_writeback,
            grad_checkpoint=self.model.grad_checkpoint,
            token_bottleneck_div=self.model.token_bottleneck_div,
            pair_bottleneck_div=self.model.pair_bottleneck_div,
            n_heads=self.model.n_heads,
            n_layers=self.model.n_layers,
            ff_expansion=self.model.ff_expansion,
            dropout=self.model.dropout,
            v_bins=int(self.learner.v_bins),
            v_min=float(self.learner.v_min),
            v_max=float(self.learner.v_max),
            use_type_embedding=self.model.use_type_embedding,
            use_triangles=self.model.use_triangles,
            use_ar_head=self.model.use_ar_head,
            ar_head_layers=self.model.ar_head_layers,
            ar_head_n_heads=self.model.ar_head_n_heads,
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
