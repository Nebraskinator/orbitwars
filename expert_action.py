"""
Vectorized rule-based "expert" for Orbit Wars.

This is the bot whose actions become BC labels during imitation
training. It consumes the assembler's already-computed observation
tensors (edge ETAs, multi-way garrison projections, per-bucket incoming
arrivals) and emits a ``[MAX_PLANETS, 2]`` ``(target, option)`` array
for the binary action space:

    OPT_SKIP     = 0
    OPT_MAX_SEND = 1

Why this file exists separately from ``heuristic_policy.py``:
    The legacy ``HeuristicPolicy`` calls the torch-backed batched
    intercept solver and is used as a self-play opponent. This
    function is pure NumPy (no torch dep) and is the one that gets
    swept by ``heuristics_sweep.py`` -- keeping them apart lets the
    sweep import without dragging in torch.

Every magic number lives on ``ExpertActionConfig`` so the sweep can
vary them without forking the function. Defaults reproduce the
original inline assembler heuristic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from orbit_obs_utils_reftrace import (
    MAX_PLANETS,
    OPT_SKIP,
    OPT_MAX_SEND,
)


@dataclass
class ExpertActionConfig:
    """Tunable knobs for ``compute_expert_action_vectorized``."""

    # Action gating
    min_send_ships: int = 2
    safety_margin: int = 1
    no_action_prob: float = 0.024114596556741304
    wait_score: float = 1.2309844729361394
    reserve_arrival_buffer: int = 1
    src_despawn_steps: int = 3

    # Frontline / potential scoring.
    potential_prod_coef: float = 6.032479981825394
    potential_proximity_coef: float = 0.9236692683685613
    potential_saturation_coef: float = 2.04362444892058
    potential_max_dist: float = 100.0
    rearguard_min_ships: int = 13
    rearguard_gradient_threshold: float = 12.521973038596219
    rearguard_score_coef: float = 0.15407262604508565
    rearguard_max_send_frac: float = 0.5875935051405107

    # Defensive (saving a falling owned planet).
    save_base_score: float = 22.68100181642289
    save_prod_bonus: float = 34.6194092390669
    save_safety_pad: int = 8

    # Despawn opportunism.
    despawn_score: float = 9993.696061615688
    target_despawn_cutoff: int = 4

    # Standard attack scoring.
    prod_power: float = 3.5
    time_power: float = 1.8767529057481072
    snipe_mult: float = 2.588286860846303
    swarm_mult: float = 1.694561092178605

    # Sync / swarm hold.
    sync_min_src_ships: int = 39
    sync_time_power: float = 2.3979023007504594
    sync_bonus_swarm: float = 2.051707453072388
    sync_bonus_solo: float = 0.5592092240649127

    # ETA validity window.
    min_intercept_t: float = 1.0
    max_intercept_t: float = 500.0

def compute_expert_action_vectorized(
    *,
    planets,
    n_p: int,
    step: int,
    ally_inc: np.ndarray,
    enemy_inc: np.ndarray,
    edge_max_tta: np.ndarray,
    edge_max_blocked: np.ndarray,
    edge_tta_max: float,
    tta_buckets: int,
    player_id: int,
    target_reserved_until: dict,
    cfg: ExpertActionConfig,
    comet_info: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Pick a per-source (target, option) action for every owned planet.

    The bulk of scoring is ``[n_p, n_p]`` NumPy ops; the only
    Python-level iteration is the post-hoc reservation assignment
    (one row at a time, argmax + bookkeeping).
    """
    action = np.zeros((MAX_PLANETS, 2), dtype=np.int64)
    action[:, 0] = np.arange(MAX_PLANETS, dtype=np.int64)
    action[:, 1] = OPT_SKIP

    if n_p <= 0:
        return action

    # Expire stale reservations.
    expired = [pid for pid, until in target_reserved_until.items() if until <= step]
    for pid in expired:
        target_reserved_until.pop(pid, None)

    # Per-planet column vectors.
    p_owner = np.fromiter(
        (int(planets[i][1]) for i in range(n_p)), dtype=np.int64, count=n_p,
    )
    p_ships = np.fromiter(
        (float(planets[i][5]) for i in range(n_p)), dtype=np.float32, count=n_p,
    )
    p_prod = np.fromiter(
        (float(planets[i][6]) for i in range(n_p)), dtype=np.float32, count=n_p,
    )
    p_pid = np.fromiter(
        (int(planets[i][0]) for i in range(n_p)), dtype=np.int64, count=n_p,
    )

    is_owned = (p_owner == player_id)
    is_neutral = (p_owner == -1)
    is_enemy = ~is_owned & ~is_neutral
    if not is_owned.any():
        return action

    # Comet despawn flags.
    src_despawn = np.zeros(n_p, dtype=bool)
    tgt_despawn = np.zeros(n_p, dtype=bool)
    if comet_info:
        for j in range(n_p):
            pid_j = int(p_pid[j])
            if pid_j in comet_info:
                _, k, plen = comet_info[pid_j]
                steps_left = max(0, plen - 1 - k)
                if steps_left <= cfg.src_despawn_steps:
                    src_despawn[j] = True
                if steps_left <= cfg.target_despawn_cutoff:
                    tgt_despawn[j] = True

    # Edge ETA / blocked matrices, denormalized to raw turns.
    t_hit_full = edge_max_tta[:n_p, :n_p] * edge_tta_max
    blocked = edge_max_blocked[:n_p, :n_p] > 0.5
    valid_eta = (
        ~blocked
        & (t_hit_full >= cfg.min_intercept_t)
        & (t_hit_full <= cfg.max_intercept_t)
    )

    # Bucket lookup for incoming arrivals.
    b_idx = np.clip(np.ceil(t_hit_full).astype(np.int64), 0, tta_buckets)
    safe_b = np.clip(b_idx, 0, tta_buckets - 1)

    ally_cumsum = np.zeros((n_p, tta_buckets + 1), dtype=np.float32)
    enemy_cumsum = np.zeros((n_p, tta_buckets + 1), dtype=np.float32)
    ally_cumsum[:, 1:] = np.cumsum(ally_inc[:n_p], axis=1)
    enemy_cumsum[:, 1:] = np.cumsum(enemy_inc[:n_p], axis=1)
    ally_arriving = np.take_along_axis(ally_cumsum, b_idx, axis=1)
    enemy_arriving = np.take_along_axis(enemy_cumsum, b_idx, axis=1)
    inc_at_b_ally = np.take_along_axis(ally_inc[:n_p], safe_b, axis=1)
    inc_at_b_enemy = np.take_along_axis(enemy_inc[:n_p], safe_b, axis=1)

    # Frontline distance + per-planet potential score.
    enemy_pair_mask = is_enemy[None, :] & ~blocked
    masked_tta = np.where(enemy_pair_mask, t_hit_full, np.inf)
    closest_enemy_tta = masked_tta.min(axis=1)
    frontline_dist = np.where(
        np.isfinite(closest_enemy_tta), closest_enemy_tta, cfg.potential_max_dist,
    )
    safe_prod = np.maximum(p_prod, 1.0)
    proximity = np.maximum(0.0, cfg.potential_max_dist - frontline_dist)
    saturation = p_ships / safe_prod
    potential_scores = (
        safe_prod * cfg.potential_prod_coef
        + proximity * cfg.potential_proximity_coef
        - saturation * cfg.potential_saturation_coef
    )

    # Score matrix. Any chosen non-skip action is now OPT_MAX_SEND.
    score = np.full((n_p, n_p), -1.0, dtype=np.float32)

    src_ships_col = p_ships[:, None]
    src_prod_col = p_prod[:, None]
    src_despawning_col = src_despawn[:, None]
    src_is_owned = is_owned[:, None]

    tgt_ships_row = p_ships[None, :]
    tgt_prod_row = p_prod[None, :]
    tgt_owned_row = is_owned[None, :]
    tgt_enemy_row = is_enemy[None, :]
    tgt_neutral_row = is_neutral[None, :]
    tgt_despawn_row = tgt_despawn[None, :]

    eye = np.eye(n_p, dtype=bool)
    pair_mask = src_is_owned & valid_eta & ~eye
    pair_mask &= ~tgt_despawn_row

    # Branch A: reinforce a falling owned planet.
    proj_def_self = (
        tgt_ships_row + tgt_prod_row * t_hit_full + ally_arriving - enemy_arriving
    )
    falling = tgt_owned_row & (proj_def_self < 0)
    required_save = (
        np.ceil(np.maximum(-proj_def_self, 0.0)).astype(np.int64) + cfg.save_safety_pad
    )
    can_save = falling & (src_ships_col >= required_save)
    save_score = cfg.save_base_score + (
        tgt_prod_row * cfg.save_prod_bonus / np.maximum(t_hit_full, 1.0)
    )
    better = pair_mask & can_save & (save_score > score)
    score = np.where(better, save_score, score)

    # Branch B: rearguard reinforcement.
    rearguard_eligible = (
        tgt_owned_row & ~falling & (src_ships_col >= cfg.rearguard_min_ships)
    )
    gradient = potential_scores[None, :] - potential_scores[:, None]
    rearguard_eligible &= gradient > cfg.rearguard_gradient_threshold
    rearguard_score = (
        cfg.rearguard_score_coef * gradient / np.maximum(t_hit_full, 1.0)
    )
    better = pair_mask & rearguard_eligible & (rearguard_score > score)
    score = np.where(better, rearguard_score, score)

    # Branch C/D: attack an enemy or neutral target.
    prod_accrual = np.where(tgt_enemy_row, tgt_prod_row * t_hit_full, 0.0)
    proj_def_atk = tgt_ships_row + prod_accrual + enemy_arriving - ally_arriving
    required_atk = (
        np.ceil(np.maximum(proj_def_atk, 0.0)).astype(np.int64) + cfg.safety_margin
    )
    can_attack = src_ships_col.astype(np.int64) > required_atk
    attack_eligible = pair_mask & (tgt_enemy_row | tgt_neutral_row)

    # D.1: source despawning soon AND can attack.
    despawn_attack = attack_eligible & src_despawning_col & can_attack
    despawn_score = cfg.despawn_score * (
        np.maximum(tgt_prod_row, 0.25) / np.maximum(t_hit_full, 1.0)
    )
    better = despawn_attack & (despawn_score > score)
    score = np.where(better, despawn_score, score)

    # D.2: standard attack.
    prod_val = np.maximum(tgt_prod_row, 0.25) ** cfg.prod_power
    time_cost = np.maximum(t_hit_full, 1.0) ** cfg.time_power
    snipe = np.where(inc_at_b_enemy > 0, cfg.snipe_mult, 1.0)
    swarm = np.where(inc_at_b_ally > 0, cfg.swarm_mult, 1.0)
    std_attack_score = (prod_val / time_cost) * snipe * swarm
    std_eligible = attack_eligible & ~src_despawning_col & can_attack
    better = std_eligible & (std_attack_score > score)
    score = np.where(better, std_attack_score, score)

    # D.3: sync / swarm hold.
    swarm_hold = (
        attack_eligible
        & ~src_despawning_col
        & ~can_attack
        & (ally_arriving > 0)
        & (src_ships_col > cfg.sync_min_src_ships)
    )
    sync_time = np.maximum(t_hit_full, 1.0) ** cfg.sync_time_power
    sync_bonus = np.where(
        inc_at_b_ally > 0, cfg.sync_bonus_swarm, cfg.sync_bonus_solo,
    )
    sync_score = sync_bonus * (prod_val / sync_time)
    better = swarm_hold & (sync_score > score)
    score = np.where(better, sync_score, score)

    # Sequential reservation assignment.
    reserved_mask = np.zeros(n_p, dtype=bool)
    for pid, until in target_reserved_until.items():
        if until <= step:
            continue
        slot = np.where(p_pid == int(pid))[0]
        if slot.size > 0:
            reserved_mask[slot[0]] = True
    if reserved_mask.any():
        score[:, reserved_mask] = -1.0

    enough_ships = (p_ships >= cfg.min_send_ships) & is_owned
    for i in np.where(enough_ships)[0]:
        if rng.random() < cfg.no_action_prob:
            continue
        row = score[i]
        if not np.any(row > cfg.wait_score):
            continue
        j = int(np.argmax(row))
        if row[j] <= cfg.wait_score:
            continue
        action[i, 0] = j
        action[i, 1] = OPT_MAX_SEND

        tgt_pid = int(p_pid[j])
        t_hit = float(t_hit_full[i, j])
        target_reserved_until[tgt_pid] = int(
            step + math.ceil(t_hit) + cfg.reserve_arrival_buffer,
        )
        score[:, j] = -1.0

    return action
