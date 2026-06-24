"""
Core Neural Network and Proximal Policy Optimization (PPO) Utilities.

This module provides the OrbitTransformer architecture for the Kaggle Orbit Wars
environment, along with high-performance sampling, GAE calculation, and PPO
update logic.

Token layout per step (Markovian, no inter-turn attention):

    [global_tok, planet_tok x MAX_PLANETS, register_tok, critic_tok]

Each planet/comet body is its own token (planets and comets share one
encoder, so the body count is bounded by MAX_PLANETS = 32 planets +
4 comets per spawn group). Fleets are *not* tokenised: their arrivals
are folded into the per-planet incoming/timeline projections inside the
observation. A single global token carries the dense scalars (turn
number, ship totals, comet timer, etc.). ``register_tok`` is a learned
aggregator the body can route information through. ``critic_tok`` is
the query for the value-head cross-attention.

Action heads are per-source-planet: an edge-aware joint head
(``EdgeActionHead``) emits ``MAX_PLANETS * N_ACTION_OPTIONS`` logits
per source -- a pointer over target planet/comet slots × a 12-option
discrete launch-size choice (see ``orbit_obs_utils.OPT_*``):

    OPT_SKIP     = 0   wait this turn (canonical label pairs SKIP with
                       ``target_idx == src_slot``)
    OPT_TAKEOVER = 1   send exactly the count needed to capture the
                       target (defenders + production * t_hit + safety),
                       resolved iteratively by the action interpreter
    OPT_PCT_10..OPT_PCT_100 (2..11)
                       send ``floor(garrison * k * 0.10)``

The interpreter in ``orbit_obs.py`` resolves the chosen target into a
firing angle by lead-targeting (closed-form for static planets, orbit
walker for orbiting planets, engine path iterator for comets) and
resolves OPT_TAKEOVER's ship count from the target's defenders and
production, so the policy never has to learn intercept geometry or the
exact-takeover arithmetic.

``ppo_update`` and ``AsyncEpisodeDataset`` keep a few generic plumbing fields
(``mb_ep_ids``, ``batch_seq_len``, a single ``action_mask`` slice) that are not
all used by the per-planet multi-discrete action path.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Iterator, Final, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
# Setup logger for core model events
logger = logging.getLogger(__name__)

def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Applies a boolean mask to logits, setting invalid actions to a large negative value.
    
    Includes a safety valve to prevent NaN outputs if an entirely zero mask is provided
    by forcing at least one valid index.
    """
    mask_sum = mask.sum(dim=-1)
    if (mask_sum == 0).any():
        bad_indices = (mask_sum == 0).nonzero(as_tuple=True)[0]
        mask = mask.clone()
        mask[bad_indices, 0] = 1.0
        logger.warning(f"Zero mask detected at batch indices {bad_indices.tolist()}. Forced index 0.")

    m = (mask > 0.5).to(torch.bool)
    return logits.masked_fill(~m, -1e4)


def twohot_targets(x: torch.Tensor, *, v_min: float, v_max: float, v_bins: int) -> torch.Tensor:
    """
    Encodes scalar values into a two-hot distribution over uniform bins.
    
    Used for distributional value prediction to reduce variance and improve 
    learning stability in reinforcement learning.
    """
    x = x.clamp(v_min, v_max)
    scale = (v_bins - 1) / (v_max - v_min)
    f = (x - v_min) * scale
    i0 = torch.floor(f).long()
    i1 = torch.clamp(i0 + 1, max=v_bins - 1)

    w1 = (f - i0.float())
    w0 = 1.0 - w1

    # Edge case: exactly at the maximum bin
    w0 = torch.where(i0 == i1, torch.ones_like(w0), w0)
    w1 = torch.where(i0 == i1, torch.zeros_like(w1), w1)

    t = torch.zeros((x.shape[0], v_bins), device=x.device, dtype=x.dtype)
    t.scatter_add_(1, i0.view(-1, 1), w0.view(-1, 1))
    t.scatter_add_(1, i1.view(-1, 1), w1.view(-1, 1))
    return t

def expand_twohot(x: torch.Tensor, v_min: float, v_max: float, v_bins: int) -> torch.Tensor:
    """Expands a tensor of shape [..., L] to [..., L * v_bins] using two-hot encoding."""
    x_clamp = x.clamp(v_min, v_max)
    scale = (v_bins - 1) / (v_max - v_min)
    f = (x_clamp - v_min) * scale
    i0 = torch.floor(f).long()
    i1 = torch.clamp(i0 + 1, max=v_bins - 1)

    w1 = f - i0.to(x.dtype)
    w0 = 1.0 - w1

    mask_max = (i0 == i1)
    w0 = torch.where(mask_max, torch.ones_like(w0), w0)
    w1 = torch.where(mask_max, torch.zeros_like(w1), w1)

    # Scatter into one-hot array
    out_shape = list(x.shape) + [v_bins]
    out = torch.zeros(out_shape, device=x.device, dtype=x.dtype)
    out.scatter_(dim=-1, index=i0.unsqueeze(-1), src=w0.unsqueeze(-1))
    out.scatter_(dim=-1, index=i1.unsqueeze(-1), src=w1.unsqueeze(-1))
    
    # Flatten the last two dimensions (L and v_bins)
    return out.flatten(start_dim=-2)

def apply_twohot_specs(tensor: torch.Tensor, specs: list) -> torch.Tensor:
    """Dynamically slices and expands a raw tensor based on schema instructions."""
    if not specs: return tensor
    chunks = []
    curr_idx = 0
    
    # Process specs strictly left-to-right based on start offset
    for spec in sorted(specs, key=lambda s: s["start"]):
        start = spec["start"]
        length = spec["length"]
        
        # Append unchanged raw data before this spec
        if start > curr_idx:
            chunks.append(tensor[..., curr_idx:start])
        
        # Expand the target slice
        slice_to_expand = tensor[..., start : start + length]
        expanded = expand_twohot(slice_to_expand, spec["min"], spec["max"], spec["bins"])
        chunks.append(expanded)
        
        curr_idx = start + length
        
    # Append any remaining raw data at the end of the tensor
    if curr_idx < tensor.shape[-1]:
        chunks.append(tensor[..., curr_idx:])
        
    return torch.cat(chunks, dim=-1)

def dist_value_loss(v_logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    """Computes cross-entropy loss between predicted value logits and target distributions."""
    logp = torch.log_softmax(v_logits, dim=-1)
    return -(target_dist * logp).sum(dim=-1).mean()

import math

@torch.no_grad()
def masked_sample(
    logits: torch.Tensor,
    mask: torch.Tensor,
    greedy: bool = False,
    temp: float = 1.0,
    top_p: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples actions from masked logits.

    Returns:
        action: sampled/argmax action
        logp:   log-prob under the actual behavior distribution used to choose action
        entropy: entropy of the unscaled policy distribution, useful for telemetry
    """
    ml_pure = masked_logits(logits, mask)
    dist_pure = Categorical(logits=ml_pure)

    if greedy:
        a = torch.argmax(ml_pure, dim=-1)
        return a, dist_pure.log_prob(a), torch.zeros_like(a, dtype=torch.float32)

    ml_explore = masked_logits(logits / max(float(temp), 1e-4), mask)

    if 0.0 < top_p < 1.0:
        probs = torch.softmax(ml_explore, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_to_remove = cumulative_probs > top_p
        sorted_to_remove[..., 1:] = sorted_to_remove[..., :-1].clone()
        sorted_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(sorted_to_remove)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_to_remove)

        ml_explore = ml_explore.masked_fill(indices_to_remove, float("-inf"))

    dist_explore = Categorical(logits=ml_explore)
    a = dist_explore.sample()

    # Important for PPO: logp_old must match the behavior policy.
    behavior_logp = dist_explore.log_prob(a)

    # Entropy telemetry can use the pure policy distribution so the reported
    # entropy reflects the model, not the exploration wrapper.
    entropy = dist_pure.entropy()

    return a, behavior_logp, entropy


def masked_logprob_entropy(
    logits: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates log probabilities and entropy for a specific set of actions under a mask."""
    ml = masked_logits(logits, mask)
    dist = Categorical(logits=ml)
    return dist.log_prob(actions), dist.entropy()


def joint_logprob_entropy(
    pi_logits: torch.Tensor,    # [B, P, P * F]
    joint_mask: torch.Tensor,   # [B, P, P * F]
    act_flat: torch.Tensor,     # [B, P]
    source_mask: torch.Tensor,  # [B, P]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-step log-prob and entropy for the joint 1D action space.
    """
    B, P, num_actions = pi_logits.shape
    
    # Flatten batch and planets to use the standard masked_logprob helper
    pi_flat = pi_logits.reshape(B * P, num_actions)
    mask_flat = joint_mask.reshape(B * P, num_actions)
    act_flat_1d = act_flat.reshape(B * P).clamp(min=0, max=num_actions - 1).long()
    
    logp_flat, ent_flat = masked_logprob_entropy(pi_flat, mask_flat, act_flat_1d)
    
    # Reshape back and mask out inactive sources
    sm = source_mask.to(logp_flat.dtype)
    logp = (logp_flat.reshape(B, P) * sm).sum(dim=-1)
    entropy = (ent_flat.reshape(B, P) * sm).sum(dim=-1)
    
    return logp, entropy

def logit_soft_cap(score, b, h, q_idx, kv_idx):
    """
    Applies soft-clipping to attention logits: 
    C * tanh(score / C)
    """
    cap = 50.0
    return torch.tanh(score / cap) * cap

# ----------------------------------------
# 2. READOUT (Q-only cross-attention; used by the actor/critic head)
# ----------------------------------------

class FlexReadout(nn.Module):
    """
    Per-turn readout attention.

    Query tokens:
      0 = actor
      1 = critic

    Key/value tokens:
      full token current turn
    """
    def __init__(self, d_model, n_heads, ff_expansion=2.0, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, int(d_model * ff_expansion)),
            nn.GELU(),
            nn.Linear(int(d_model * ff_expansion), d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        q_x:                [B, Q, D]  actor/critic query tokens
        kv_x:               [B, K, D]  full current-step token sequence
        key_padding_mask:   [B, K] bool, True = valid (will attend), False = pad
        """
        B, Q, D = q_x.shape
        K = kv_x.shape[1]

        q = self.q_proj(self.norm_q(q_x)).view(B, Q, self.n_heads, -1).transpose(1, 2)
        k = self.k_proj(self.norm_kv(kv_x)).view(B, K, self.n_heads, -1).transpose(1, 2)
        v = self.v_proj(self.norm_kv(kv_x)).view(B, K, self.n_heads, -1).transpose(1, 2)

        # Build SDPA attention mask from key padding. SDPA expects a bool mask
        # broadcastable to [B, H, Q, K] where True = attend.
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.view(B, 1, 1, K).expand(B, self.n_heads, Q, K)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        q_out = q_x + self.out_proj(attn_out.transpose(1, 2).reshape(B, Q, D))
        return q_out + self.ff(self.norm_ff(q_out))

# ----------------------------------------
# 3. CONTINUOUS POSITION / SPATIAL EMBEDDING
# ----------------------------------------

class ContinuousFourierEmbedding(nn.Module):
    """
    Projects continuous coordinates into high-dimensional sine/cosine wave bands.
    This enables transformers to process high-fidelity geometric distance.
    """
    def __init__(self, num_coords: int = 4, num_freqs: int = 16):
        super().__init__()
        self.num_coords = num_coords
        self.num_freqs = num_freqs
        
        # Log-linear frequency bands from 2^0 to 2^8
        # Broad bands capture board quadrants; tight bands capture local precision
        freqs = 2.0 ** torch.linspace(0.0, 8.0, num_freqs)
        self.register_buffer("frequencies", freqs * math.pi)
        
        # Output dim: 4 coordinates * 16 bands * 2 (sin+cos) = 128 dimensions
        self.out_dim = num_coords * num_freqs * 2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., num_coords]
        x_proj = x.unsqueeze(-1) * self.frequencies  # [..., num_coords, num_freqs]
        x_sin = torch.sin(x_proj)
        x_cos = torch.cos(x_proj)
        # [..., num_coords * num_freqs * 2]
        return torch.cat([x_sin, x_cos], dim=-1).flatten(start_dim=-2)
    
# ----------------------------------------
# 4. ORBIT TRANSFORMER
# ----------------------------------------

# System tokens after the entity tokens. Layout per step:
#   [global_tok, planet_toks, fleet_toks, register_tok, critic_tok]
# (No actor token: the policy is per-planet, so action logits come from the
# contextualized planet tokens themselves -- there's nothing for a single
# global "actor" query to do.)
N_SYSTEM_TOKENS = 3   # global + register + critic

# ----------------------------------------
# 4A. EDGE-AWARE NODE/PAIR/TRIANGLE MODULES
# ----------------------------------------

class SmallMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairBiasedPlanetAttention(nn.Module):
    """
    Planet self-attention with directed edge/pair bias.

    For source/query i and target/key j:

        logit_ij = q_i dot k_j / sqrt(d) + bias(z_ij)

    where z_ij is the learned edge state initialized from edge_features[i, j].
    """

    def __init__(self, d_model: int, pair_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.pair_dim = pair_dim
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = float(dropout)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_scale = nn.Parameter(torch.full((n_heads, 1, 1), 0.5))
        self.pair_scale = nn.Parameter(torch.full((n_heads, 1, 1), 1.))

        self.pair_bias_norm = nn.LayerNorm(pair_dim)
        # Evoformer-style bias projection: Linear(LayerNorm(z)) -> per-head bias.
        # The downstream softmax provides the nonlinearity, so a hidden layer
        # here adds parameters without adding capacity. See Algorithm 7 step 5.
        self.pair_to_bias = nn.Linear(pair_dim, n_heads)

        # Evoformer-style output gate (Algorithm 7, step 6): a per-position,
        # per-channel sigmoid gate that lets each token selectively admit
        # attention output before the final projection. This gives positions
        # a switch to dampen attention contributions that are dominated by
        # the pair_bias term, which would otherwise be forced through.
        # Initialized in _reset_parameters to ~0.88 (bias=2.0, weight=0) so
        # behavior at init is near the pre-gate version.
        self.gate_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        Z: torch.Tensor,             # [B, S, D]
        z: torch.Tensor,             # [B, P, P, E]
        valid_mask: torch.Tensor,    # [B, S]
        p_start: int,
        p_end: int,
    ) -> torch.Tensor:

        B, S, D = Z.shape
        H = self.n_heads
        Hd = self.head_dim

        q = self.q_proj(Z).view(B, S, H, Hd).transpose(1, 2)
        k = self.k_proj(Z).view(B, S, H, Hd).transpose(1, 2)
        v = self.v_proj(Z).view(B, S, H, Hd).transpose(1, 2)

        # 1. Scale Q and K directly for the head-specific attn_scale
        # (Using sqrt so Q * K^T effectively multiplies the logit by the scale)
        safe_a_scale = torch.clamp(
            self.attn_scale.float().view(1, H, 1, 1),
            min=1e-4,
        ).to(dtype=q.dtype, device=q.device)
        q_scaled = q * torch.sqrt(safe_a_scale)
        k_scaled = k * torch.sqrt(safe_a_scale)

        # 2. Build the padded edge bias
        pair_bias = self.pair_to_bias(self.pair_bias_norm(z)).permute(0, 3, 1, 2)
        pad_left = p_start
        pad_right = S - p_end
        pad_top = p_start
        pad_bottom = S - p_end
        
        # F.pad handles the zero-expansion efficiently
        full_bias = F.pad(pair_bias, (pad_left, pad_right, pad_top, pad_bottom))
        
        # 3. Scale the edge bias
        p_scale = self.pair_scale.float().view(1, H, 1, 1).to(
            dtype=full_bias.dtype,
            device=full_bias.device,
        )
        scaled_full_bias = full_bias * p_scale

        # 4. Merge the validity mask into the bias as -inf
        if valid_mask is not None:
            key_valid = valid_mask.view(B, 1, 1, S)
            scaled_full_bias = scaled_full_bias.masked_fill(~key_valid, float('-inf'))

        # 5. Let the optimized PyTorch C++ backend work
        out = F.scaled_dot_product_attention(
            q_scaled, 
            k_scaled, 
            v,
            attn_mask=scaled_full_bias,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )

        out = out.transpose(1, 2).contiguous().view(B, S, D)

        # Evoformer-style gating: o_si <- g_si * o_si (per-position, per-channel).
        # Gate is computed from the pre-attention token state Z (matches Alg. 7).
        gate = torch.sigmoid(self.gate_proj(Z))
        out = out * gate

        out = self.out_proj(out)

        if valid_mask is not None:
            out = out * valid_mask[:, :, None].to(out.dtype)

        return out


class TriangleMultiplicativeUpdate(nn.Module):
    """
    Protein-folding-style triangle update over directed pair states.

    outgoing:
        z_ij uses z_ik and z_jk

    incoming:
        z_ij uses z_ki and z_kj
    """

    def __init__(self, pair_dim: int, mode: str, bottleneck_dim):
        super().__init__()
        assert mode in {"outgoing", "incoming"}

        self.mode = mode
        self.norm = nn.LayerNorm(pair_dim)

        self.left_proj = nn.Linear(pair_dim, bottleneck_dim)
        self.right_proj = nn.Linear(pair_dim, bottleneck_dim)

        self.left_gate = nn.Sequential(nn.Linear(pair_dim, bottleneck_dim), nn.Sigmoid())
        self.right_gate = nn.Sequential(nn.Linear(pair_dim, bottleneck_dim), nn.Sigmoid())
        self.out_gate = nn.Sequential(nn.Linear(pair_dim, pair_dim), nn.Sigmoid())

        self.out_proj = nn.Linear(bottleneck_dim, pair_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # Start triangle path as identity. This prevents the new O(P^3) path
        # from destabilizing the model at initialization.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        z: torch.Tensor,             # [B, P, P, E]
        pair_mask: torch.Tensor,     # [B, P, P] bool
    ) -> torch.Tensor:

        B, P, _, E = z.shape

        zn = self.norm(z)

        left = self.left_proj(zn) * self.left_gate(zn)
        right = self.right_proj(zn) * self.right_gate(zn)

        if pair_mask is not None:
            m = pair_mask[..., None].to(left.dtype)
            left = left * m
            right = right * m

        if self.mode == "outgoing":
            # update[i, j] = sum_k left[i, k] * right[j, k]
            update = torch.einsum("bikc,bjkc->bijc", left, right)
        else:
            # update[i, j] = sum_k left[k, i] * right[k, j]
            update = torch.einsum("bkic,bkjc->bijc", left, right)

        update = update / math.sqrt(max(P, 1))
        update = self.out_proj(update)
        update = update * self.out_gate(zn)

        if pair_mask is not None:
            update = update * pair_mask[..., None].to(update.dtype)

        return z + update


class NodePairTriangleBlock(nn.Module):
    """
    One node/pair block:

      1. node update with pair-biased attention
      2. pair update from [z_ij, h_i, h_j, raw_edge_ij]
      3. outgoing/incoming triangle updates
      4. pair feedforward transition
    """

    def __init__(
        self,
        d_model: int,
        pair_dim: int,
        edge_dim: int,
        n_heads: int,
        token_bottleneck_div: float = 4.0,
        pair_bottleneck_div: float = 4.0,
        ff_expansion: float = 2.0,
        dropout: float = 0.0,
        use_triangles: bool = True,
    ):
        super().__init__()

        self.use_triangles = bool(use_triangles)
        self.token_bottleneck_dim = int(d_model//token_bottleneck_div)
        self.attn_scale = nn.Parameter(torch.tensor(1.0))

        self.node_norm1 = nn.LayerNorm(d_model)
        self.node_attn = PairBiasedPlanetAttention(
            d_model=d_model,
            pair_dim=pair_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.node_norm2 = nn.LayerNorm(d_model)
        self.node_ff = nn.Sequential(
            nn.Linear(d_model, int(d_model * ff_expansion)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * ff_expansion), d_model),
            nn.Dropout(dropout),
        )

        self.pair_norm1 = nn.LayerNorm(pair_dim)
        self.node_bottleneck = nn.Linear(d_model, self.token_bottleneck_dim)
        
        self.pair_update = SmallMLP(
            in_dim=pair_dim + edge_dim + (2 * self.token_bottleneck_dim), # 2 * 16 instead of 2 * 144
            hidden_dim=d_model,
            out_dim=pair_dim,
            dropout=dropout,
        )

        if self.use_triangles:
            self.tri_out = TriangleMultiplicativeUpdate(pair_dim, mode="outgoing", bottleneck_dim=int(pair_dim//pair_bottleneck_div))
            self.tri_in = TriangleMultiplicativeUpdate(pair_dim, mode="incoming", bottleneck_dim=int(pair_dim//pair_bottleneck_div))
        else:
            self.register_module('tri_out', None)
            self.register_module('tri_in', None)

        self.pair_norm2 = nn.LayerNorm(pair_dim)
        self.pair_ff = nn.Sequential(
            nn.Linear(pair_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pair_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        Z: torch.Tensor,             # [B, S, D]
        z: torch.Tensor,             # [B, P, P, E]
        edge_raw: torch.Tensor,      # [B, P, P, Ed]
        valid_mask: torch.Tensor,    # [B, S]
        edge_mask: torch.Tensor,     # [B, P, P]
        p_start: int,
        p_end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, S, D = Z.shape
        P = p_end - p_start

        planet_mask = valid_mask[:, p_start:p_end]
        pm = planet_mask.bool()
        pair_mask = pm[:, :, None] & pm[:, None, :] & edge_mask.bool()

        # 1. Full Sequence Node update
        attn_resid_scale = self.attn_scale.float().to(dtype=Z.dtype, device=Z.device)
        Z = Z + self.node_attn(self.node_norm1(Z), z, valid_mask, p_start, p_end) * attn_resid_scale
        Z = Z + self.node_ff(self.node_norm2(Z))
        Z = Z * valid_mask[:, :, None].to(Z.dtype)

        # 2. Extract planet tokens for Pair update
        h_planets = Z[:, p_start:p_end, :]
        h_compressed = self.node_bottleneck(h_planets) # [B, P, bottleneck_dim]
        
        src_h = h_compressed[:, :, None, :].expand(B, P, P, self.token_bottleneck_dim)
        tgt_h = h_compressed[:, None, :, :].expand(B, P, P, self.token_bottleneck_dim)

        pair_in = torch.cat(
            [
                self.pair_norm1(z),
                src_h,
                tgt_h,
                edge_raw,
            ],
            dim=-1,
        )

        z = z + self.pair_update(pair_in)
        z = z * pair_mask[..., None].to(z.dtype)

        # 3. Triangle updates
        if self.use_triangles:
            z = self.tri_out(z, pair_mask)
            z = self.tri_in(z, pair_mask)

        # 4. Pair transition
        z = z + self.pair_ff(self.pair_norm2(z))
        z = z * pair_mask[..., None].to(z.dtype)

        return Z, z


class EdgeActionHead(nn.Module):
    """
    Edge-first action head.

    For every directed edge i -> j:

        edge_ctx_ij = MLP([source_token_i, target_token_j, pair_state_ij])

    Predicts:
        target_score_ij       [B, P, P]
        option_logits_ijk     [B, P, P, F]

    Joint logits:
        joint_logits_ijk = target_score_ij + option_logits_ijk

    Flattened output remains [B, P, P * F], so PPO/BC code stays compatible.
    """

    def __init__(self, d_model: int, pair_dim: int, n_options: int, hidden_dim: int, bottleneck_div: float = 4.0):
        super().__init__()

        self.n_options = int(n_options)
        self.bottleneck_dim = int(d_model//bottleneck_div)
        self.bottleneck_div = bottleneck_div
        if bottleneck_div > 1.0:
            self.src_proj = nn.Linear(d_model, self.bottleneck_dim)
            self.tgt_proj = nn.Linear(d_model, self.bottleneck_dim)
        else:
            self.src_proj = nn.Identity()
            self.tgt_proj = nn.Identity()
        
        in_dim = (2 * self.bottleneck_dim) + pair_dim
        self.edge_norm = nn.LayerNorm(in_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.option_head = nn.Linear(hidden_dim, self.n_options)

        self.reset_parameters()

    def reset_parameters(self):
        # Width-aware (He) init: std = sqrt(2 / fan_in). This keeps activation
        # variance roughly stable through GELU/ReLU-style nonlinearities and
        # avoids disproportionately under-initializing narrow streams (e.g.
        # the pair channel) the way a fixed std=0.02 does.
        if self.bottleneck_div > 1.0:
            for m in [self.src_proj, self.tgt_proj]:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=math.sqrt(2.0 / m.in_features))
                    nn.init.zeros_(m.bias)

        for m in self.edge_mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=math.sqrt(2.0 / m.in_features))
                nn.init.zeros_(m.bias)

        # Start uniform after masking.
        #nn.init.zeros_(self.target_score_head.weight)
        #nn.init.zeros_(self.target_score_head.bias)
        nn.init.zeros_(self.option_head.weight)
        nn.init.zeros_(self.option_head.bias)

    def forward(
        self,
        planet_h: torch.Tensor,      # [B, P, D]
        pair_z: torch.Tensor,        # [B, P, P, E]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, P, D = planet_h.shape

        src = self.src_proj(planet_h)
        tgt = self.tgt_proj(planet_h)

        src_b = src[:, :, None, :].expand(B, P, P, self.bottleneck_dim)
        tgt_b = tgt[:, None, :, :].expand(B, P, P, self.bottleneck_dim)

        edge_in = torch.cat([src_b, tgt_b, pair_z], dim=-1)
        edge_in = self.edge_norm(edge_in)

        ctx = self.edge_mlp(edge_in)

        option_logits = self.option_head(ctx)

        joint_logits = option_logits.reshape(B, P, P * self.n_options)
        
        dummy_target_score = torch.zeros(B, P, P, device=planet_h.device, dtype=planet_h.dtype)

        return joint_logits, dummy_target_score, option_logits
    
class OrbitTransformer(nn.Module):
    """
    Single-step transformer over a heterogeneous token set:
    one global token, MAX_PLANETS planet/comet tokens (shared encoder),
    plus two learned aggregator tokens (register, critic) at the tail.
    Fleets are *not* tokenised -- their arrivals are folded into the
    per-planet incoming/timeline projections inside the observation.

    Action is per-planet factored multi-discrete ``(target_idx, option_idx)``:
    each owned source planet picks a target *planet/comet* slot and one of
    ``N_ACTION_OPTIONS`` (= 12) discrete launch sizes -- one SKIP, one
    iterative exact-TAKEOVER, and 10 fixed percentage buckets (10%..100%
    of garrison). The action interpreter (in ``orbit_obs.py``) computes
    the firing angle by lead-targeting the chosen target body and
    resolves the TAKEOVER ship count from the target's defenders and
    production, so the policy never has to learn intercept geometry or
    exact-takeover arithmetic. See ``orbit_obs_utils.OPT_*`` for the
    full option table.

    Heads:
      - policy: edge-aware joint head over planet tokens
        (``EdgeActionHead``), output ``MAX_PLANETS * N_ACTION_OPTIONS``
        logits per source. Joint layout: ``action = target * F + option``.
      - value:  cross-attention readout from the critic token.
    """

    def __init__(
        self,
        meta: dict,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_expansion: float = 2.0,
        dropout: float = 0.0,
        v_bins: int = 101,
        v_min: float = -5,
        v_max: float = 5,
        use_type_embedding: bool = True,
        inference_only: bool = False,
        **kwargs,
    ):
        super().__init__()
        # `meta` is the dict returned by `orbit_obs.schema_metadata()`.
        self.meta = meta
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.inference_only = inference_only

        self.max_planets = int(meta["max_planets"])
        self.planet_dim = int(meta["planet_dim"])
        self.global_dim = int(meta["global_dim"])
        self.edge_dim = int(meta["edge_dim"])
        self.pair_dim = int(kwargs.get("pair_dim", 128))
        self.use_triangles = bool(kwargs.get("use_triangles", True))
        self.token_bottleneck_div = float(kwargs.get("token_bottleneck_div", 2.0))
        self.pair_bottleneck_div = float(kwargs.get("pair_bottleneck_div", 2.0))
        # Per-target option count.
        # Binary action space:
        #   0 = wait
        #   1 = max-send
        self.n_options = int(meta["n_action_options"])
        self.action_dim_per_planet = self.max_planets * self.n_options

        # Total token count per step.
        self.seq_len = self.max_planets + N_SYSTEM_TOKENS

        # Reusable index slices into the contextualized sequence.
        self.idx_global = 0
        self.idx_planet_start = 1
        self.idx_planet_end = 1 + self.max_planets
        self.idx_register = self.idx_planet_end
        self.idx_critic = self.idx_register + 1
        assert self.idx_critic + 1 == self.seq_len

        # ---- Per-entity-type subnets (input -> d_model) ----
        self.num_freqs = 8
        self.spatial_emb = ContinuousFourierEmbedding(num_coords=4, num_freqs=self.num_freqs)
        
        self.planet_twohot_specs = meta.get("planet_twohot_specs", [])
        self.edge_twohot_specs = meta.get("edge_twohot_specs", [])
        self.global_twohot_specs = meta.get("global_twohot_specs", [])
        # Raw edge-feature offsets from schema_metadata().
        # Used for action masking before PPO samples.
        self.edge_offsets = dict(meta.get("edge_offsets", {}))
        self.edge_off_max_blocked = int(self.edge_offsets.get("max_blocked", 1))
        
        self.expanded_planet_dim = self.planet_dim
        for spec in self.planet_twohot_specs:
            self.expanded_planet_dim += spec["length"] * spec["bins"] - spec["length"]
            
        self.expanded_edge_dim = self.edge_dim
        for spec in self.edge_twohot_specs:
            self.expanded_edge_dim += spec["length"] * spec["bins"] - spec["length"]
            
        self.expanded_global_dim = self.global_dim
        for spec in self.global_twohot_specs:
            self.expanded_global_dim += spec["length"] * spec["bins"] - spec["length"]

        # Calculate new input sizes: (Expanded Dim - 4 raw floats) + 128 Fourier dimensions
        p_in_dim = (self.expanded_planet_dim - 4) + self.spatial_emb.out_dim
        
        self.planet_net = self._build_subnet(p_in_dim, d_model)
        self.global_net = self._build_subnet(self.expanded_global_dim, d_model)

        # ---- Optional type embedding (planet=0, fleet=1, system=2) ----
        self.use_type_embedding = use_type_embedding
        if use_type_embedding:
            self.type_emb = nn.Embedding(3, d_model)
            type_ids = torch.empty(self.seq_len, dtype=torch.long)
            type_ids[self.idx_global] = 1
            type_ids[self.idx_planet_start:self.idx_planet_end] = 0
            type_ids[self.idx_register] = 1
            type_ids[self.idx_critic] = 1
            self.register_buffer("type_ids", type_ids, persistent=False)
        
        self.planet_slot_emb = nn.Embedding(self.max_planets, d_model)

        # ---- Aggregator tokens ----
        self.register_tok = nn.Parameter(torch.randn(1, 1, d_model))
        self.critic_tok = nn.Parameter(torch.randn(1, 1, d_model))

        # ---- Transformer body ----
        # ---- Edge-aware node/pair trunk ----
        self.global_to_planet = nn.Linear(d_model, d_model)
        
        self.edge_net = self._build_subnet(self.expanded_edge_dim, self.pair_dim)
        
        self.node_pair_trunk = nn.ModuleList(
            [
                NodePairTriangleBlock(
                    d_model=d_model,
                    pair_dim=self.pair_dim,
                    edge_dim=self.expanded_edge_dim,
                    n_heads=n_heads,
                    ff_expansion=ff_expansion,
                    dropout=dropout,
                    use_triangles=self.use_triangles,
                    token_bottleneck_div=self.token_bottleneck_div,
                    pair_bottleneck_div=self.pair_bottleneck_div,
                )
                for _ in range(n_layers)
            ]
        )

        # Pre-LN final norms (GPT-2 `ln_f`), one per stream. The per-block
        # pre-norms inside NodePairTriangleBlock keep each sublayer's *input*
        # bounded, which is what gives Pre-LN its O(1/sqrt(L)) gradient
        # behavior; that design intentionally lets the residual stream itself
        # grow through depth. These terminal norms bound the trunk *output*
        # before the heads consume it. Important for v_head (a bare Linear
        # with no internal pre-norm) and for keeping the three sub-channels
        # in EdgeActionHead.edge_norm (src, tgt, pair_z) on comparable scales
        # after concatenation.
        self.final_node_norm = nn.LayerNorm(d_model)
        self.final_pair_norm = nn.LayerNorm(self.pair_dim)

        self.readout = FlexReadout(
            d_model,
            n_heads,
            ff_expansion=ff_expansion,
            dropout=dropout,
        )
        
        # ---- Heads ----
        self.pi_head = EdgeActionHead(
            d_model=d_model,
            pair_dim=self.pair_dim,
            n_options=self.n_options,
            hidden_dim=max(self.pair_dim, d_model),
            bottleneck_div=1.,
        )
        
        self.v_head = nn.Linear(d_model, v_bins)

        # ---- JEPA auxiliary (kept structurally; action conditioning removed
        # since Orbit Wars uses a multi-discrete per-planet action). ----
        if not self.inference_only:
            self.jepa_target_emb = nn.Embedding(self.max_planets + 1, d_model)
            self.jepa_frac_emb = nn.Embedding(self.n_options + 1, d_model)
            
            self.jepa_action_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            
            self.jepa_predictor = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )
            
            # Edge/pair JEPA: predicts next-turn pair_z from current pair_z.
            self.jepa_pair_action_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, self.pair_dim),
                nn.GELU(),
                nn.Linear(self.pair_dim, self.pair_dim),
            )
            
            self.jepa_pair_predictor = nn.Sequential(
                nn.LayerNorm(self.pair_dim),
                nn.Linear(self.pair_dim, self.pair_dim * 2),
                nn.GELU(),
                nn.Linear(self.pair_dim * 2, self.pair_dim),
            )
            
            self.psi_predictor = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )
            
            self.psi_pair_predictor = nn.Sequential(
                nn.LayerNorm(self.pair_dim),
                nn.Linear(self.pair_dim, self.pair_dim * 2),
                nn.GELU(),
                nn.Linear(self.pair_dim * 2, self.pair_dim),
            )

        self.unpacker = OrbitObservationUnpacker(meta)
        self.register_buffer("v_support", torch.linspace(v_min, v_max, v_bins))
        self._reset_parameters()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _build_subnet(self, in_d: int, out_d: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_d, out_d * 2),
            nn.GELU(),
            nn.Linear(out_d * 2, out_d),
            nn.LayerNorm(out_d),
        )

    def _reset_parameters(self):
        # Width-aware (He) init for Linear: std = sqrt(2 / fan_in). A fixed
        # std=0.02 (GPT-2 convention) disproportionately under-initializes
        # narrow streams: Linear(64,64) ends up at ~0.11x He, Linear(256,256)
        # at ~0.23x. That penalizes the pair channel when pair_dim is small.
        # Embeddings stay at std=0.02 -- He doesn't apply (fan_in is 1 for
        # one-hot lookup).
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=math.sqrt(2.0 / m.in_features))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        # Depth-aware residual scaling factor (GPT-2 / Megatron pattern):
        # 1/sqrt(2L) scales residual-exit projections so the residual stream
        # variance stays roughly bounded at init regardless of depth. The 2
        # in 2L accounts for two residual contributions per block (attention
        # and FF). Applied unconditionally now -- the old `n_layers > 7`
        # threshold was too lenient; at L=5 the residual stream already
        # accumulates Var(Z) ~ 11 without scaling, which biases gradient
        # flow toward early layers and triggers per-layer collapse downstream.
        resid_scale = 1.0 / math.sqrt(2.0 * self.n_layers)
        std_proj = 0.02 * resid_scale

        # Readout cross-attention: spiky Q/K, standard V, depth-scaled residual exits.
        nn.init.xavier_uniform_(self.readout.q_proj.weight, gain=1.5)
        nn.init.xavier_uniform_(self.readout.k_proj.weight, gain=1.5)
        nn.init.normal_(self.readout.v_proj.weight, std=0.02)
        nn.init.normal_(self.readout.out_proj.weight, std=std_proj)
        nn.init.normal_(self.readout.ff[2].weight, std=std_proj)

        for block in self.node_pair_trunk:
            # 1. Small non-zero init on the edge bias projection. Zero-init
            #    makes the bias bootstrap rate scale with pair_dim (each
            #    gradient step gives bias magnitude ~ E * lr * grad), so
            #    narrow pair_dim takes proportionally longer to develop a
            #    useful bias signal. A small std=0.01 gives the bias a tiny
            #    structured starting point without overwhelming the standard
            #    SDPA path. Note: pair_to_bias is now a single nn.Linear
            #    (no .net[-1] indexing).
            nn.init.normal_(block.node_attn.pair_to_bias.weight, std=0.01)
            nn.init.zeros_(block.node_attn.pair_to_bias.bias)

            # 2. Zero the pair update so edge residuals start as identity
            nn.init.zeros_(block.pair_update.net[-1].weight)
            nn.init.zeros_(block.pair_update.net[-1].bias)

            # 3. Zero the pair transition FF (index 3 is the final Linear layer)
            nn.init.zeros_(block.pair_ff[3].weight)
            nn.init.zeros_(block.pair_ff[3].bias)

            # 4. Output gate on node attention: init weight=0, bias=2.0 so
            #    sigmoid(bias) ~ 0.88 at start. This preserves near-pre-gate
            #    attention magnitude (only ~12% dampening) so existing
            #    training dynamics aren't shocked by suddenly halving
            #    attention output (which would happen with default bias=0).
            #    The model can learn to lower the gate where attention isn't
            #    useful (e.g., when pair_bias is the dominant routing signal).
            nn.init.zeros_(block.node_attn.gate_proj.weight)
            nn.init.constant_(block.node_attn.gate_proj.bias, 2.0)

            # 5. Depth-aware scaling on the trunk's residual-exit projections.
            #    Each block writes into the token residual stream via
            #    node_attn.out_proj and node_ff[3]. Without scaling, the
            #    residual stream variance grows ~2L through depth (one
            #    attention + one FF residual per block), starving deep-layer
            #    gradients and biasing routing toward early layers. Scaling
            #    these exits by 1/sqrt(2L) (from He init) keeps the residual
            #    stream variance bounded at init. Pair-side exits
            #    (pair_update.net[-1], pair_ff[3]) are already zero-init and
            #    don't need this scaling.
            nn.init.normal_(
                block.node_attn.out_proj.weight,
                std=math.sqrt(2.0 / block.node_attn.out_proj.in_features) * resid_scale,
            )
            nn.init.zeros_(block.node_attn.out_proj.bias)
            nn.init.normal_(
                block.node_ff[3].weight,
                std=math.sqrt(2.0 / block.node_ff[3].in_features) * resid_scale,
            )
            nn.init.zeros_(block.node_ff[3].bias)
            
        # Keep triangle update initially near identity.
        for m in self.modules():
            if isinstance(m, TriangleMultiplicativeUpdate):
                m.reset_parameters()
    
        nn.init.normal_(self.register_tok, std=0.02)
        nn.init.normal_(self.critic_tok, std=0.02)
    
        self.pi_head.reset_parameters()
    
        nn.init.zeros_(self.v_head.weight)
        nn.init.zeros_(self.v_head.bias)

    # ------------------------------------------------------------------
    # Parameter zoning (for the Learner's per-zone LR/grad governance)
    # ------------------------------------------------------------------

    def get_parameter_zones(self) -> Dict[str, List[Tuple[str, nn.Parameter]]]:
        zones = {
            "subnets": [], "transformer": [], "readout": [],
            "jepa": [], "pi": [], "v": [], "embeddings": [],
        }
        
        jepa_prefixes = (
            "jepa_predictor",
            "jepa_target_emb",
            "jepa_frac_emb",
            "jepa_action_proj",
            "jepa_pair_predictor",
            "jepa_pair_action_proj",
            "psi_predictor",   
            "psi_pair_predictor",
        )

        for name, p in self.named_parameters():
            if name.startswith("pi_head"):
                zones["pi"].append((name, p))
            elif name.startswith("v_head"):
                zones["v"].append((name, p))
            elif name.startswith("readout"):
                zones["readout"].append((name, p))
            elif name.startswith(jepa_prefixes):
                zones["jepa"].append((name, p))
            elif name.startswith(("type_emb", "planet_slot_emb")):
                zones["embeddings"].append((name, p))
            elif name.startswith(("planet_net.", "global_net.", "edge_net.", "global_to_planet.")):
                zones["subnets"].append((name, p))
            else:
                # Everything else safely maps to the main body relational graph
                zones["transformer"].append((name, p))
                
        return zones

    def enable_bf16_recurrent_path(self):
        bf16_modules = [
            self.planet_net,
            self.global_net,
            self.global_to_planet,
            self.edge_net,
            self.node_pair_trunk,
            self.final_node_norm,
            self.final_pair_norm,
            self.readout,
            self.pi_head,
            self.v_head,
        ]
        
        if not self.inference_only:
            bf16_modules.extend([
                self.jepa_predictor,
                self.jepa_action_proj,
                self.jepa_target_emb,
                self.jepa_frac_emb,
                self.jepa_pair_predictor,
                self.jepa_pair_action_proj,
                self.psi_predictor, 
                self.psi_pair_predictor,
            ])
    
        if self.use_type_embedding:
            bf16_modules.append(self.type_emb)
    
        bf16_modules.append(self.planet_slot_emb)
    
        for mod in bf16_modules:
            mod.bfloat16()
    
        self.register_tok.data = self.register_tok.data.bfloat16()
        self.critic_tok.data = self.critic_tok.data.bfloat16()
        self.keep_sensitive_params_fp32()
        
    def _force_param_fp32(self, module: nn.Module, name: str) -> None:
        """
        Re-register a named parameter as FP32.
    
        Must be called before torch.compile() and before optimizer construction.
        """
        p = getattr(module, name)
        if not isinstance(p, nn.Parameter):
            raise TypeError(f"{module}.{name} is not an nn.Parameter")
    
        setattr(
            module,
            name,
            nn.Parameter(
                p.detach().float().clone(),
                requires_grad=p.requires_grad,
            ),
        )
    
    
    def keep_sensitive_params_fp32(self):
        """
        Must be called at the end of enable_bf16_recurrent_path(),
        before torch.compile() and before optimizer construction.
        """
        for block in self.node_pair_trunk:
            self._force_param_fp32(block, "attn_scale")
            self._force_param_fp32(block.node_attn, "attn_scale")
            self._force_param_fp32(block.node_attn, "pair_scale")

    def recurrent_dtype(self) -> torch.dtype:
        return self.critic_tok.dtype

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    
    def encode_jepa_actions(
        self,
        act: torch.Tensor,          # [B, P, 2], target + frac
        source_mask: torch.Tensor,  # [B, P]
    ) -> torch.Tensor:
        """
        Returns an action-conditioning tensor shaped [B, 1 + P, D],
        aligned to [global_token, planet_tokens].

        It includes:
          - outgoing action embedding on the source planet
          - incoming action embedding scattered onto the target planet
          - global summary action embedding on the global token
        """
        b_dim, p_dim, _ = act.shape
        d_model = self.d_model
        dev = act.device

        act_target = act[..., 0].clamp(min=0, max=self.max_planets - 1).long()
        act_frac = act[..., 1].clamp(min=0, max=self.n_options - 1).long()

        sm = source_mask.to(dtype=torch.float32, device=dev)
        launched = sm * (act_frac > 0).to(sm.dtype)

        # 1. outbound action embedding (source -> out)
        a_out = self.jepa_target_emb(act_target) + self.jepa_frac_emb(act_frac)
        a_out = a_out * launched.unsqueeze(-1).to(a_out.dtype)  # [b_dim, p_dim, d_model]

        # 2. inbound action embedding (scattered to targets)
        a_in = torch.zeros_like(a_out)
        scatter_idx = act_target.unsqueeze(-1).expand(b_dim, p_dim, d_model)
        
        # use scatter_add_ so multiple incoming fleets targeting the same planet sum their embeddings
        a_in.scatter_add_(dim=1, index=scatter_idx, src=a_out)

        # 3. combine them for the planet node conditioning
        planet_action = a_out + a_in  # [b_dim, p_dim, d_model]

        # 4. global summary
        denom = launched.sum(dim=1, keepdim=True).clamp_min(1.0)  # [b_dim, 1]
        global_action = a_out.sum(dim=1, keepdim=True) / denom.unsqueeze(-1)

        action_facets = torch.cat([global_action, planet_action], dim=1)  # [b_dim, 1+p_dim, d_model]
        return self.jepa_action_proj(action_facets.to(self.recurrent_dtype()))
    
    def encode_jepa_edge_actions(
        self,
        act: torch.Tensor,          # [B, P, 2], target + option
        source_mask: torch.Tensor,  # [B, P]
    ) -> torch.Tensor:
        """
        Returns [B, P, P, pair_dim], aligned to directed edge i -> j.
    
        If source i launches to target j, edge_action[:, i, j] receives
        an embedding of that chosen target/option. Skips contribute zero.
        """
    
        B, P, _ = act.shape
        dev = act.device
    
        act_target = act[..., 0].clamp(min=0, max=self.max_planets - 1).long()
        act_frac = act[..., 1].clamp(min=0, max=self.n_options - 1).long()
    
        launched = (
            source_mask.to(dtype=torch.float32, device=dev)
            * (act_frac > 0).to(torch.float32)
        )
    
        # Reuse existing action embeddings.
        a = self.jepa_target_emb(act_target) + self.jepa_frac_emb(act_frac)
        a = a * launched.unsqueeze(-1).to(a.dtype)  # [B, P, D]
    
        # Scatter source action embedding onto selected directed edge i -> target_i.
        edge_action = torch.zeros(
            B,
            P,
            P,
            self.d_model,
            device=dev,
            dtype=a.dtype,
        )
    
        scatter_idx = act_target.view(B, P, 1, 1).expand(B, P, 1, self.d_model)
        edge_action.scatter_(dim=2, index=scatter_idx, src=a.unsqueeze(2))
    
        return self.jepa_pair_action_proj(edge_action.to(self.recurrent_dtype()))
    
    def encode_features(self, obs_flat: torch.Tensor):
        obs = self.unpacker(obs_flat)
        B = obs_flat.shape[0]
        rec_dtype = self.recurrent_dtype()
    
        p_in = obs["planet_features"].to(rec_dtype)
        
        p_expanded = apply_twohot_specs(p_in, self.planet_twohot_specs)
        p_spatial, p_rest = p_expanded[..., :4], p_expanded[..., 4:]
    
        p_fourier = self.spatial_emb(p_spatial).to(rec_dtype)
        p_flat = torch.cat([p_fourier, p_rest], dim=-1).view(
            -1,
            p_fourier.shape[-1] + p_rest.shape[-1],
        )
    
        p_tokens = self.planet_net(p_flat).view(
            B,
            self.max_planets,
            self.d_model,
        )
    
        #planet_slots = torch.arange(self.max_planets, device=obs_flat.device)
        #p_tokens = p_tokens + self.planet_slot_emb(planet_slots).to(p_tokens.dtype).unsqueeze(0)
    
        g_in = obs["global_features"].to(rec_dtype)
        g_expanded = apply_twohot_specs(g_in, self.global_twohot_specs)
        global_tok = self.global_net(g_expanded).unsqueeze(1)
    
        edge_raw = obs["edge_features"].to(rec_dtype)
        edge_features = apply_twohot_specs(edge_raw, self.edge_twohot_specs)
        
        edge_mask = obs["edge_mask"].to(rec_dtype)
    
        return global_tok, p_tokens, obs, edge_features, edge_mask

    def _build_valid_mask(self, planet_mask: torch.Tensor) -> torch.Tensor:
        B = planet_mask.shape[0]
        valid = torch.ones(B, self.seq_len, dtype=torch.bool, device=planet_mask.device)
        valid[:, self.idx_planet_start:self.idx_planet_end] = planet_mask.bool()
        return valid

    def condition_tokens(
        self,
        global_tok: torch.Tensor,
        p_tokens: torch.Tensor,
        obs: Dict[str, torch.Tensor],
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Z_t:    [B, 1 + P + 2, D]
            pair_z: [B, P, P, pair_dim]
        """
    
        B = global_tok.shape[0]
        planet_mask = obs["planet_mask"]
    
        # Let global state condition all planet tokens.
        pair_z = self.edge_net(edge_features.reshape(-1, self.expanded_edge_dim)).view(
            B, self.max_planets, self.max_planets, self.pair_dim,
        )
    
        pm = planet_mask.bool()
        pair_mask = pm[:, :, None] & pm[:, None, :] & edge_mask.bool()
        pair_z = pair_z * pair_mask[..., None].to(pair_z.dtype)
    
        # Construct full sequence upfront
        Z_t = torch.cat(
            [
                global_tok,
                p_tokens,
                self.register_tok.expand(B, 1, -1),
                self.critic_tok.expand(B, 1, -1),
            ],
            dim=1,
        )
    
        if self.use_type_embedding:
            Z_t = Z_t + self.type_emb(self.type_ids).to(Z_t.dtype).unsqueeze(0)
    
        valid = self._build_valid_mask(obs["planet_mask"])
        Z_t = Z_t * valid[:, :, None].to(Z_t.dtype)
        
        for layer in self.node_pair_trunk:
            Z_t, pair_z = layer(
                Z=Z_t,
                z=pair_z,
                edge_raw=edge_features,
                valid_mask=valid,
                edge_mask=edge_mask,
                p_start=self.idx_planet_start,
                p_end=self.idx_planet_end,
            )

        # Pre-LN terminal normalization. Bounds the trunk output before the
        # heads consume it; per-block pre-norms already handle inputs to each
        # sublayer, so internal residual-stream growth is the intended Pre-LN
        # behavior. LayerNorm doesn't preserve zero-padding, so re-apply the
        # validity masks after normalizing.
        Z_t = self.final_node_norm(Z_t)
        pair_z = self.final_pair_norm(pair_z)
        Z_t = Z_t * valid[:, :, None].to(Z_t.dtype)
        pair_z = pair_z * pair_mask[..., None].to(pair_z.dtype)

        return Z_t, pair_z
    
    # ------------------------------------------------------------------
    # Heads
    # ------------------------------------------------------------------

    def build_joint_mask(
        self,
        planet_mask: torch.Tensor,
        source_mask: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Builds a [B, P, P * F] joint validity mask.

        Binary action space:
            action = target_idx * F + option_idx
            option 0 = wait
            option 1 = max-send

        Source rows:
          - wait is legal only at (target == source, option == 0)
          - max-send is legal only to valid, non-self, unblocked target slots

        Non-source rows:
          - only canonical wait is legal

        edge_features should be the RAW edge feature tensor from obs_dict,
        not the expanded/two-hot edge tensor.
        """
        B, P = planet_mask.shape
        F = int(self.n_options)
        device = planet_mask.device

        tgt_valid = planet_mask.unsqueeze(1).expand(B, P, P).to(torch.float32)
        src_valid = source_mask.unsqueeze(-1).expand(B, P, P).to(torch.float32)

        eye = torch.eye(P, device=device, dtype=torch.float32)
        eye_b = eye.unsqueeze(0).expand(B, P, P)
        nonself = 1.0 - eye_b

        if edge_mask is None:
            edge_valid = torch.ones(B, P, P, device=device, dtype=torch.float32)
        else:
            edge_valid = edge_mask.to(torch.float32)

        if edge_features is None:
            unblocked = torch.ones(B, P, P, device=device, dtype=torch.float32)
        else:
            blocked = edge_features[..., self.edge_off_max_blocked] > 0.5
            unblocked = (~blocked).to(torch.float32)

        # Max-send is legal only for owned source planets targeting valid,
        # non-self, unblocked body slots.
        real_target_mask = tgt_valid * src_valid * nonself * edge_valid * unblocked

        joint_mask = torch.zeros(B, P, P, F, device=device, dtype=torch.float32)

        # Canonical wait: always legal on the diagonal. This prevents fully
        # zero rows even for padded/non-source rows.
        joint_mask[..., 0] = eye_b

        # Binary max-send option.
        if F > 1:
            joint_mask[..., 1] = real_target_mask

        return joint_mask.reshape(B, P, P * F)

    def readout_policy(
        self,
        Z_t: torch.Tensor,
        pair_z: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        read_q = Z_t[:, self.idx_critic:self.idx_critic + 1, :]
        readout = self.readout(read_q, Z_t, key_padding_mask=valid_mask)
    
        planet_z = Z_t[:, self.idx_planet_start:self.idx_planet_end, :]
    
        pi_logits, target_score, option_logits = self.pi_head(
            planet_z,
            pair_z,
        )
    
        v_logits = self.v_head(readout[:, 0, :])
        v_exp = (torch.softmax(v_logits, dim=-1) * self.v_support).sum(dim=-1)
    
        return (
            pi_logits.float(),
            v_logits.float(),
            v_exp.float(),
            target_score.float(),
            option_logits.float(),
        )

    def forward(self, obs_flat: torch.Tensor, return_edge_scores: bool = False):
        global_tok, p_tokens, obs, edge_features, edge_mask = self.encode_features(obs_flat)
    
        Z_t, pair_z = self.condition_tokens(
            global_tok=global_tok,
            p_tokens=p_tokens,
            obs=obs,
            edge_features=edge_features,
            edge_mask=edge_mask,
        )
    
        valid_mask = self._build_valid_mask(obs["planet_mask"])
    
        pi_logits, v_logits, v_exp, target_score, option_logits = self.readout_policy(
            Z_t=Z_t,
            pair_z=pair_z,
            valid_mask=valid_mask,
        )
    
        joint_mask = self.build_joint_mask(
            obs["planet_mask"],
            obs["source_mask"],
            edge_features=obs["edge_features"],
            edge_mask=edge_mask,
        )
    
        if return_edge_scores:
            return pi_logits, v_logits, v_exp, joint_mask, target_score, option_logits
    
        return pi_logits, v_logits, v_exp, joint_mask

    def forward_jepa(self, obs_flat: torch.Tensor, return_pair: bool = False):
        global_tok, p_tokens, obs, edge_features, edge_mask = self.encode_features(obs_flat)
    
        Z_t, pair_z = self.condition_tokens(
            global_tok=global_tok,
            p_tokens=p_tokens,
            obs=obs,
            edge_features=edge_features,
            edge_mask=edge_mask,
        )
    
        state_facets_t = Z_t[:, self.idx_global:self.idx_planet_end, :]
        Z_pred = self.jepa_predictor(state_facets_t)
    
        if return_pair:
            return Z_pred, Z_t, pair_z
    
        return Z_pred, Z_t


class OrbitObservationUnpacker(nn.Module):
    """
    Slices a flat observation tensor into the structured Orbit Wars components.

    Flat layout:

        [global_features (G),
         planet_features (P * Pd),
         edge_features   (P * P * Ed),
         edge_mask       (P * P),
         planet_mask     (P),
         source_mask     (P)]

    Returns:
        global_features: [B, G]
        planet_features: [B, P, Pd]
        edge_features:   [B, P, P, Ed]
        edge_mask:       [B, P, P]
        planet_mask:     [B, P]
        source_mask:     [B, P]
    """

    def __init__(self, meta: dict):
        super().__init__()
        self.meta = meta

        self.max_planets = int(meta["max_planets"])
        self.planet_dim = int(meta["planet_dim"])
        self.global_dim = int(meta["global_dim"])
        self.edge_dim = int(meta["edge_dim"])

        s = 0
        offsets: Dict[str, Tuple[int, int]] = {}

        offsets["global_features"] = (s, s + self.global_dim)
        s += self.global_dim

        offsets["planet_features"] = (s, s + self.max_planets * self.planet_dim)
        s += self.max_planets * self.planet_dim

        offsets["edge_features"] = (s, s + self.max_planets * self.max_planets * self.edge_dim)
        s += self.max_planets * self.max_planets * self.edge_dim

        offsets["edge_mask"] = (s, s + self.max_planets * self.max_planets)
        s += self.max_planets * self.max_planets

        offsets["planet_mask"] = (s, s + self.max_planets)
        s += self.max_planets

        offsets["source_mask"] = (s, s + self.max_planets)
        s += self.max_planets

        self.offsets = offsets
        self.total_dim = s

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {
            name: x[:, start:end]
            for name, (start, end) in self.offsets.items()
        }

        out["planet_features"] = out["planet_features"].reshape(
            -1, self.max_planets, self.planet_dim
        )

        out["edge_features"] = out["edge_features"].reshape(
            -1, self.max_planets, self.max_planets, self.edge_dim
        )

        out["edge_mask"] = out["edge_mask"].reshape(
            -1, self.max_planets, self.max_planets
        ).to(torch.float32)

        out["planet_mask"] = out["planet_mask"].to(torch.float32)
        out["source_mask"] = out["source_mask"].to(torch.float32)

        return out


def pack_observation(
    global_features: torch.Tensor,   # [B, G] or [G]
    planet_features: torch.Tensor,   # [B, P, Pd] or [P, Pd]
    edge_features: torch.Tensor,     # [B, P, P, Ed] or [P, P, Ed]
    edge_mask: torch.Tensor,         # [B, P, P] or [P, P]
    planet_mask: torch.Tensor,       # [B, P] or [P]
    source_mask: torch.Tensor,       # [B, P] or [P]
) -> torch.Tensor:
    """
    Flatten the assembler dict into a single tensor matching
    OrbitObservationUnpacker's layout.

    Returns:
        [B, total_dim]
    """

    if global_features.dim() == 1:
        global_features = global_features.unsqueeze(0)

    if planet_features.dim() == 2:
        planet_features = planet_features.unsqueeze(0)

    if edge_features.dim() == 3:
        edge_features = edge_features.unsqueeze(0)

    if edge_mask.dim() == 2:
        edge_mask = edge_mask.unsqueeze(0)

    if planet_mask.dim() == 1:
        planet_mask = planet_mask.unsqueeze(0)

    if source_mask.dim() == 1:
        source_mask = source_mask.unsqueeze(0)

    B = global_features.shape[0]

    parts = [
        global_features,
        planet_features.reshape(B, -1),
        edge_features.reshape(B, -1),
        edge_mask.reshape(B, -1),
        planet_mask,
        source_mask,
    ]

    return torch.cat(parts, dim=1)

# ----------------------------
# Training Components
# ----------------------------

@dataclass
class PPOUpdateStats:
    """Statistics container for a single PPO update batch."""
    approx_kl: float
    clip_frac: float

    # Normalized entropy: average entropy per active source planet.
    entropy: float

    v_loss: float
    pg_loss: float
    jepa_loss: float
    total_loss: float
    n_mb: int

class AsyncEpisodeDataset:
    def __init__(self, act_dim: int, device: str):
        self.act_dim, self.device = act_dim, device
        self.clear()

    def __len__(self): return int(self.n_steps)

    def clear(self):
        # Flattened storage for performance
        self.obs, self.act, self.logp, self.val, self.adv, self.ret, self.next_hp = [], [], [], [], [], [], []
        self.ep_ids = []  # New: tracks boundaries via flat IDs
        self.n_steps = 0
        self.ep_counter = 0 
        self._tensor_cache = None

    def add_steps(self, obs, act, logp, val, adv, ret, next_hp, ep_ids):
        """
        Maintains the flat plumbing. 'ep_ids' should match the length of the steps.
        """
        def dev(x): return x.detach().to(self.device)
        self.obs.append(obs.detach().to("cpu", non_blocking=True))
        self.act.append(dev(act)); self.logp.append(dev(logp))
        self.val.append(dev(val)); self.adv.append(dev(adv))
        self.ret.append(dev(ret)); self.next_hp.append(dev(next_hp))
        self.ep_ids.append(dev(ep_ids))
        self.n_steps += int(act.shape[0])

    def tensorize(self) -> Tuple:
        if self._tensor_cache: return self._tensor_cache
        self._tensor_cache = (
            torch.cat(self.obs, dim=0), torch.cat(self.act, dim=0), 
            torch.cat(self.logp, dim=0).float(), torch.cat(self.val, dim=0).float(), 
            torch.cat(self.adv, dim=0).float(), torch.cat(self.ret, dim=0).float(), 
            torch.cat(self.next_hp, dim=0).float(),
            torch.cat(self.ep_ids, dim=0) # Index 7: Episode Boundaries
        )
        return self._tensor_cache
    
    def swap_out_tensor_cache(self):
        data = self.tensorize(); self.clear(); return data
        
    def _episode_bounds(self, ep_ids: torch.Tensor) -> list[tuple[int, int]]:
        n = int(ep_ids.numel())
        if n == 0:
            return []

        changes = torch.nonzero(ep_ids[1:] != ep_ids[:-1], as_tuple=False).flatten() + 1
        bounds = torch.cat([
            torch.tensor([0], dtype=torch.long, device=ep_ids.device),
            changes,
            torch.tensor([n], dtype=torch.long, device=ep_ids.device),
        ])
        return [(int(bounds[i]), int(bounds[i + 1])) for i in range(bounds.numel() - 1)]

    def iter_minibatches(
        self, mb_size: int, shuffle_episodes: bool = False, shuffle_steps: bool = False
    ) -> Iterator[Tuple]:
        # Sequence-Aware or Flat Slicing
        data = self.tensorize()
        ep_ids = data[7]
        n = int(ep_ids.shape[0])
        
        if shuffle_steps:
            # Full flat shuffle (destroys t -> t+1 ordering, ideal for pure PPO/BC)
            order = torch.randperm(n, dtype=torch.long)
        elif shuffle_episodes:
            # Chunk-level shuffle (preserves t -> t+1 ordering within episodes for JEPA)
            spans = self._episode_bounds(ep_ids)
            perm = torch.randperm(len(spans)).tolist()
            order = torch.cat([
                torch.arange(spans[i][0], spans[i][1], dtype=torch.long)
                for i in perm
            ], dim=0)
        else:
            order = torch.arange(n, dtype=torch.long)
            
        usable = (int(order.numel()) // mb_size) * mb_size
        if usable == 0:
            return
        
        order = order[:usable]
        for start in range(0, usable, mb_size):
            idx = order[start:start + mb_size]
            yield tuple(d[idx] for d in data)

@torch.no_grad()
def gae_from_episode(
    rewards: torch.Tensor,     # [T]
    values: torch.Tensor,      # [T]
    dones: torch.Tensor,       # [T] (1 at terminal step)
    gamma: float,
    lam: float,
    last_value: float = 0.0,   # terminal -> 0
    variances: Optional[torch.Tensor] = None, # [T] State-conditional variance
    lam_min: float = 0.55,
    lam_max: float = 0.95,
    lam_eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes GAE for one episode. If variances are provided, computes state-aware 
    dynamic lambda per timestep to balance bias and variance dynamically.
    """
    T = rewards.shape[0]
    adv = torch.zeros((T,), device=rewards.device)
    gae = torch.tensor(0.0, device=rewards.device)

    # ---------------------------------------------------------
    # Branch A: Dynamic Lambda calculation using per-state Variance
    # ---------------------------------------------------------
    if variances is not None:
        # Precompute 1-step TD Errors for the entire episode
        next_values = torch.cat([values[1:], torch.tensor([last_value], device=values.device)])
        next_nonterminal = 1.0 - dones
        
        deltas = rewards + gamma * next_values * next_nonterminal - values
        
        # Vectorized dynamic lambda calculation
        delta_sq = deltas ** 2
        dynamic_lambda = delta_sq / (delta_sq + variances + lam_eps)
        dynamic_lambda = torch.clamp(dynamic_lambda, min=lam_min, max=lam_max)

        for t in reversed(range(T)):
            gae = deltas[t] + gamma * dynamic_lambda[t] * next_nonterminal[t] * gae
            adv[t] = gae

    # ---------------------------------------------------------
    # Branch B: Standard Static GAE
    # ---------------------------------------------------------
    else:
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - dones[t]
            next_value = torch.tensor(last_value, device=rewards.device) if (t == T - 1) else values[t + 1]
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            gae = delta + gamma * lam * next_nonterminal * gae
            adv[t] = gae

    ret = adv + values
    return adv, ret

def masked_smoothed_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, smoothing: float = 0.01) -> torch.Tensor:
    """
    Computes cross-entropy loss with label smoothing applied ONLY to legal actions.

    Illegal actions are forced to 0.0 probability in the target distribution.
    """
    target_idx = targets.unsqueeze(1).to(torch.int64)
    mask = mask.clone()
    mask.scatter_(1, target_idx, 1.0)
    mask = mask.to(logits.dtype)

    ml = masked_logits(logits, mask)
    log_probs = F.log_softmax(ml, dim=-1)

    with torch.no_grad():
        legal_counts = mask.sum(dim=-1, keepdim=True)
        smooth_prob = smoothing / legal_counts
        target_dist = mask * smooth_prob
        confidence_mass = torch.full_like(
            targets.unsqueeze(1), 1.0 - smoothing, dtype=logits.dtype
        )
        target_idx = targets.unsqueeze(1).to(torch.int64)
        target_dist.scatter_add_(1, target_idx, confidence_mass)

    loss = -(target_dist * log_probs).sum(dim=-1).mean()
    return loss

def sampled_pair_jepa_loss(
    pair_pred_t: torch.Tensor,       # [B-1, P, P, E]
    pair_targ_t1: torch.Tensor,      # [B-1, P, P, E]
    edge_mask_t: torch.Tensor,       # [B-1, P, P]
    edge_mask_t1: torch.Tensor,      # [B-1, P, P]
    valid_trans: torch.Tensor,       # [B-1]
    max_pairs: int = 32768,
) -> torch.Tensor:
    """
    Smooth-L1 edge JEPA loss over valid same-episode t -> t+1 pairs.

    We subsample valid edges because [B, P, P, pair_dim] can be large.
    """

    device = pair_pred_t.device
    E = pair_pred_t.shape[-1]

    pair_valid = (
        valid_trans[:, None, None].bool()
        & edge_mask_t.bool()
        & edge_mask_t1.bool()
    )

    flat_valid = pair_valid.reshape(-1)
    idx = torch.nonzero(flat_valid, as_tuple=False).flatten()

    if idx.numel() == 0:
        return pair_pred_t.sum() * 0.0

    if max_pairs is not None and max_pairs > 0 and idx.numel() > max_pairs:
        perm = torch.randperm(idx.numel(), device=device)[:max_pairs]
        idx = idx[perm]

    pred_flat = pair_pred_t.reshape(-1, E).index_select(0, idx)
    targ_flat = pair_targ_t1.reshape(-1, E).index_select(0, idx)

    return F.smooth_l1_loss(pred_flat.float(), targ_flat.float())

def ppo_update(
    net: nn.Module,
    opt: optim.Optimizer,
    dataset: AsyncEpisodeDataset,
    scheduler=None,
    mode: str = "ppo",
    target_net: Optional[nn.Module] = None,
    **cfg,
) -> PPOUpdateStats:
    """
    PPO update for OrbitTransformer's per-planet (target, frac) action.

    Single-step Markovian forward (no batch_seq_len, no episode packing).
    Action storage: ``mb_act`` is ``[B, MAX_PLANETS, 2]`` (target, frac).
    ``mb_logp_old`` is per-step scalar (sum over active sources).
    """
    dev = next(net.parameters()).device
    stats = {k: 0.0 for k in ["kl", "clip", "ent", "v", "pg", "jepa", "total"]}
    n_mb = 0
    stop_training = False

    grad_accum_steps = int(cfg.get("grad_accum_steps", 1))
    P = int(net.max_planets)
    src_start, src_end = net.unpacker.offsets["source_mask"]
    needs_sequence = "jepa" in mode
    for epoch in range(int(cfg.get("update_epochs", 1))):
        epoch_kl, epoch_mb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        
        i = -1
        for i, mb in enumerate(dataset.iter_minibatches(
                int(cfg["minibatch_size"]), shuffle_episodes=needs_sequence,
                shuffle_steps=not needs_sequence)):
            (mb_obs, mb_act, mb_logp_old, _mb_val,
             mb_adv, mb_ret, _mb_next_hp, mb_ep) = mb

            mb_obs = mb_obs.to(dev, non_blocking=True)
            mb_act = mb_act.to(dev)
            mb_logp_old = mb_logp_old.to(dev).float()
            mb_adv = mb_adv.to(dev).float()
            mb_ret = mb_ret.to(dev).float()
            mb_ep = mb_ep.to(dev)

            act_target = mb_act[..., 0]
            act_option = mb_act[..., 1]
            
            # Flatten (target, option) into one joint action index.
            act_flat = (act_target * net.n_options) + act_option
            mb_source_mask = mb_obs[:, src_start:src_end].to(torch.float32)

            # --- SINGLE SHARED FORWARD PASS ---
            # 1. Compute the heavy transformer backbone (Z_t) exactly once
            global_tok, p_tokens, obs_dict, edge_features, edge_mask = net.encode_features(mb_obs)

            Z_t, pair_z = net.condition_tokens(
                global_tok=global_tok,
                p_tokens=p_tokens,
                obs=obs_dict,
                edge_features=edge_features,
                edge_mask=edge_mask,
            )
            
            valid_mask = net._build_valid_mask(obs_dict["planet_mask"])
            
            pi_logits, v_logits, _v_exp, _target_score, _option_logits = net.readout_policy(
                Z_t=Z_t,
                pair_z=pair_z,
                valid_mask=valid_mask,
            )
            joint_mask = net.build_joint_mask(
                obs_dict["planet_mask"],
                obs_dict["source_mask"],
                edge_features=obs_dict["edge_features"],
                edge_mask=edge_mask,
            )

            target_dist_v = twohot_targets(
                mb_ret, v_min=cfg["v_min"], v_max=cfg["v_max"], v_bins=cfg["v_bins"]
            )
            v_loss = dist_value_loss(v_logits, target_dist_v)

            # --- JEPA Auxiliary Loss ---
            jepa_loss = torch.tensor(0.0, device=dev)
            if "jepa" in mode and target_net is not None:
                is_td_jepa = "td_jepa" in mode
                # -------------------------------
                # Node/token JEPA
                # -------------------------------
                state_facets_t = Z_t[:, net.idx_global:net.idx_planet_end, :]
                # [B, 1+P, D]
            
                action_facets_t = net.encode_jepa_actions(
                    mb_act,
                    mb_source_mask,
                )
                # [B, 1+P, D]
            
                if is_td_jepa:
                    Z_pred = net.psi_predictor(state_facets_t + action_facets_t)
                else:
                    Z_pred = net.jepa_predictor(state_facets_t + action_facets_t)
            
                # -------------------------------
                # Edge/pair JEPA
                # -------------------------------
                edge_action_t = net.encode_jepa_edge_actions(
                    mb_act,
                    mb_source_mask,
                )
                # [B, P, P, pair_dim]
            
                if is_td_jepa:
                    pair_pred = net.psi_pair_predictor(pair_z + edge_action_t)
                else:
                    pair_pred = net.jepa_pair_predictor(pair_z + edge_action_t)
                # [B, P, P, pair_dim]
            
                with torch.no_grad():
                    g_tok, p_toks, targ_obs_dict, targ_edge_features, targ_edge_mask = target_net.encode_features(mb_obs)
            
                    Z_target, target_pair_z = target_net.condition_tokens(
                        global_tok=g_tok,
                        p_tokens=p_toks,
                        obs=targ_obs_dict,
                        edge_features=targ_edge_features,
                        edge_mask=targ_edge_mask,
                    )
                    
                    if is_td_jepa:
                        targ_action_facets = target_net.encode_jepa_actions(mb_act, mb_source_mask)
                        targ_state_facets = Z_target[:, net.idx_global:net.idx_planet_end, :]
                        target_psi = target_net.psi_predictor(targ_state_facets + targ_action_facets)

                        targ_edge_action = target_net.encode_jepa_edge_actions(mb_act, mb_source_mask)
                        target_pair_psi = target_net.psi_pair_predictor(target_pair_z + targ_edge_action)
            
                # Same-episode transition mask.
                valid_trans = mb_ep[:-1] == mb_ep[1:]
            
                if valid_trans.any():
                    # ---------------------------
                    # Node t -> t+1
                    # ---------------------------
                    pred_t = Z_pred[:-1, :net.idx_planet_end, :]
                    targ_t1_z = Z_target[1:, :net.idx_planet_end, :]
            
                    if is_td_jepa:
                        gamma = cfg.get("gamma", 0.997)
                        
                        # Find if t+1 is non-terminal so we can zero out future expectations
                        nonterminal_t1 = torch.cat([valid_trans[1:], torch.tensor([False], device=dev)])
                        
                        nonterm_node = nonterminal_t1.view(-1, 1, 1).float()
                        targ_t1 = targ_t1_z + gamma * target_psi[1:, :net.idx_planet_end, :] * nonterm_node
                        
                        nonterm_edge = nonterminal_t1.view(-1, 1, 1, 1).float()
                        targ_pair_t1 = target_pair_z[1:] + gamma * target_pair_psi[1:] * nonterm_edge
                    else:
                        targ_t1 = targ_t1_z
                        targ_pair_t1 = target_pair_z[1:]
            
                    node_jepa_loss = F.smooth_l1_loss(
                        pred_t[valid_trans].float(),
                        targ_t1[valid_trans].float(),
                    )
            
                    # ---------------------------
                    # Edge t -> t+1
                    # ---------------------------
                    edge_jepa_loss = sampled_pair_jepa_loss(
                        pair_pred_t=pair_pred[:-1],
                        pair_targ_t1=target_pair_z[1:],
                        edge_mask_t=obs_dict["edge_mask"][:-1],
                        edge_mask_t1=targ_obs_dict["edge_mask"][1:],
                        valid_trans=valid_trans,
                        max_pairs=int(cfg.get("edge_jepa_max_pairs", 32768)),
                    )
            
                    jepa_loss = (
                        node_jepa_loss
                        + float(cfg.get("edge_jepa_coef", 0.05)) * edge_jepa_loss
                    )
                else:
                    jepa_loss = pair_z.sum() * 0.0

            # --- Mode Routing ---
            if mode in ("warmup", "warmup_with_actor_reset"):
                pg_loss = ent_loss = approx_kl = clip_frac = torch.tensor(0.0, device=dev)
                loss = v_loss

            elif mode in ("jepa_pretraining", "td_jepa_pretraining"):
                v_loss = jepa_loss # Routed to V slot for tracking
                pg_loss = ent_loss = approx_kl = clip_frac = torch.tensor(0.0, device=dev)
                loss = jepa_loss

            elif mode in ("imitation", "imitation_frozen_backbone", "imitation_with_jepa", "imitation_with_td_jepa"):
                B = mb_obs.shape[0]
                pi_flat = pi_logits.reshape(B * P, -1).float()
                jm_flat = joint_mask.reshape(B * P, -1)
                a_flat  = act_flat.reshape(B * P).clamp(min=0, max=pi_flat.shape[-1] - 1).long()

                with torch.no_grad():
                    legal_label_flat = jm_flat.gather(-1, a_flat.unsqueeze(-1)).squeeze(-1) > 0.5
                    active_flat = mb_source_mask.reshape(B * P) > 0.5
                    '''
                    illegal_active = active_flat & (~legal_label_flat)
                    illegal_frac = illegal_active.float().mean()
                    
                    if illegal_active.any():
                        logger.warning(
                            f"Imitation batch has illegal active expert labels: "
                            f"{int(illegal_active.sum().item())}/{int(active_flat.sum().item())} "
                            f"active-source labels illegal; illegal_frac={float(illegal_frac.item()):.4f}"
                        )
                    '''
                    # Drop illegal active labels from the imitation signal
                    # entirely. The chosen logit is masked to ``-1e4`` inside
                    # ``masked_logits``, so its log-prob is roughly ``-1e4``
                    # and the per-slot CE would be ~+1e4. With the dynamic
                    # WAIT/SENDMAX weighting below (up to 100x) a handful of
                    # such slots would dominate the batch loss and push the
                    # *legal* logits down (the gradient of the masked logit
                    # itself is zero, so the only path the optimizer can take
                    # is to shrink the legal mass). Removing them is honest
                    # about the missing signal.
                    trainable_flat = active_flat & legal_label_flat
                    trainable_mask = trainable_flat.reshape(B, P).to(mb_source_mask.dtype)

                logp_joint = F.log_softmax(masked_logits(pi_flat, jm_flat), dim=-1)
                ce = -logp_joint.gather(-1, a_flat.unsqueeze(-1)).squeeze(-1).reshape(B, P)

                # Dynamically balance classes (WAIT vs SENDMAX) over the
                # *trainable* source planets only. Including illegal labels in
                # the class counts skews the dynamic weight (illegal labels are
                # nearly always OPT_MAX_SEND, since OPT_SKIP is always legal
                # on the diagonal), which would shrink the SENDMAX upweight.
                is_real_action = (act_option > 0).float()

                num_real = (is_real_action * trainable_mask).sum()
                num_active = trainable_mask.sum()
                num_wait = num_active - num_real

                # Inverse frequency weighting: (count_wait / count_real)
                # Clamped to [1.0, 100.0] to prevent explosions on sparse minibatches
                dynamic_weight = (num_wait / num_real.clamp(min=1.0)).clamp(min=1.0, max=100.0).detach()

                weights = torch.where(is_real_action > 0.5, dynamic_weight, 1.0)
                ce = ce * weights
                # ---------------------------

                # Use the same trainable mask as both the per-slot weight and
                # the denominator so the per-source average isn't quietly
                # rescaled when a few illegal labels are removed.
                denom = trainable_mask.sum().clamp(min=1.0)
                pg_loss = (trainable_mask * ce).sum() / denom
                ent_loss = approx_kl = clip_frac = torch.tensor(0.0, device=dev)

                loss = pg_loss + (cfg.get("vf_coef", 0.5) * v_loss) + (cfg.get("jepa_coef", 1.0) * jepa_loss)

            else:  # "ppo", "ppo_frozen_backbone", "ppo_with_jepa"
                logp_new, ent_sum = joint_logprob_entropy(
                    pi_logits, joint_mask, act_flat, mb_source_mask
                )
                
                ratio = (logp_new - mb_logp_old).exp()
                pg_unclipped = -mb_adv * ratio
                pg_clipped   = -mb_adv * ratio.clamp(
                    1.0 - cfg["clip_coef"],
                    1.0 + cfg["clip_coef"],
                )
                pg_loss = torch.max(pg_unclipped, pg_clipped).mean()
                
                # ent_sum is currently [B], summed over all active source planets.
                # Normalize it so entropy is "per active source planet", not "per board".
                active_sources = mb_source_mask.sum(dim=-1).clamp_min(1.0)  # [B]
                ent_loss = (ent_sum / active_sources).mean()
                
                loss = (
                    pg_loss + cfg["vf_coef"] * v_loss
                    - cfg["ent_coef"] * ent_loss
                    + cfg.get("jepa_coef", 1.0) * jepa_loss
                )
                approx_kl = (mb_logp_old - logp_new).mean()
                clip_frac = ((ratio - 1.0).abs() > cfg["clip_coef"]).float().mean()

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            if (i + 1) % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
                opt.step()
                opt.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            n_mb += 1
            epoch_mb += 1
            stats["kl"]    += float(approx_kl.detach().item())
            epoch_kl       += float(approx_kl.detach().item())
            stats["clip"]  += float(clip_frac.detach().item())
            ent_val = ent_loss.detach().item() if torch.is_tensor(ent_loss) else float(ent_loss)
            stats["ent"]   += float(ent_val)
            stats["v"]     += float(v_loss.detach().item())
            stats["pg"]    += float(pg_loss.detach().item())
            stats["jepa"]  += float(jepa_loss.detach().item())
            stats["total"] += float(loss.detach().item())

            if (epoch > 0 and mode.startswith("ppo") and cfg.get("target_kl")
                and (epoch_kl / max(epoch_mb, 1)) > cfg["target_kl"] * 1.5):
                logger.info(
                    f"Early stop at Epoch {epoch} due to KL {epoch_kl/epoch_mb:.4f}"
                )
                stop_training = True
                break

        if not stop_training and i >= 0 and (i + 1) % grad_accum_steps != 0:
            nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        if stop_training:
            break

    return PPOUpdateStats(
        approx_kl=stats["kl"] / max(n_mb, 1),
        clip_frac=stats["clip"] / max(n_mb, 1),
        entropy=stats["ent"] / max(n_mb, 1),
        v_loss=stats["v"] / max(n_mb, 1),
        pg_loss=stats["pg"] / max(n_mb, 1),
        jepa_loss=stats["jepa"] / max(n_mb, 1),
        total_loss=stats["total"] / max(n_mb, 1),
        n_mb=n_mb,
    )
