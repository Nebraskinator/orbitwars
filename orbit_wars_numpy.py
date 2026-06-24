"""
Numpy mirror of ``orbit_wars_vectorized.py``.

Same vectorized algorithm, same masks, same scatter/topk ordering -- just
with ``numpy`` in place of ``torch``. Used by the test suite to verify
algorithmic correctness in environments where torch isn't installed, and
serves as a useful CPU-only reference for embedded systems.

The two files are kept structurally identical (function names, control
flow, mask names) so changes to one can be mirrored mechanically into
the other. Diff them as part of any algorithmic change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


BOARD_SIZE = 100.0
SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS = 10.0
SHIP_SPEED_MAX = 6.0
ORBITAL_LIMIT = 50.0
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)
DEFAULT_EPISODE_STEPS = 500
NEUTRAL = -1

_EPS = 1e-9
_LOG_1000 = math.log(1000.0)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def fleet_speed_np(ships):
    f = ships.astype(np.float32)
    f_clamped = np.maximum(f, 1.0)
    s = np.log(f_clamped) / _LOG_1000
    s_clamped = np.clip(s, 0.0, 1.0)
    speed = 1.0 + (SHIP_SPEED_MAX - 1.0) * np.power(s_clamped, 1.5)
    speed = np.where(f >= 1000.0, SHIP_SPEED_MAX, speed)
    speed = np.where(f <= 0.0, 0.0, speed)
    return speed.astype(np.float32)


def segment_enters_moving_circle_np(p0, p1, c0, c1, R, valid_mask):
    """Moving-disc collision: smallest t in [0,1] where segment p0->p1
    enters the disc whose center moves linearly c0->c1 with radius R.
    Used for orbiting planets that interpolate continuously. For comets
    pass c0==c1==new_pos to fall back to static."""
    v = (p1 - p0) - (c1 - c0)
    r = p0 - c0
    a = (v * v).sum(-1)
    b = 2.0 * (v * r).sum(-1)
    c0_quad = (r * r).sum(-1) - R * R
    inside_start = c0_quad <= 0.0
    disc = b * b - 4.0 * a * c0_quad
    has_real = disc >= 0.0
    sd = np.sqrt(np.maximum(disc, 0.0))
    a_safe = a + _EPS
    t1 = (-b - sd) / (2.0 * a_safe)
    INF = np.full_like(t1, np.inf)
    t1_in = np.where((t1 >= 0.0) & (t1 <= 1.0), t1, INF)
    t1_in = np.where(has_real, t1_in, INF)
    t1_in = np.where(inside_start, 0.0, t1_in)
    t1_in = np.where(valid_mask, t1_in, INF)
    t1_in = np.where((a < _EPS) & (~inside_start), INF, t1_in)
    return t1_in


def segment_enters_circle_np(p0, p1, c, R, valid_mask):
    v = p1 - p0
    r = p0 - c
    a = (v * v).sum(-1)
    b = 2.0 * (v * r).sum(-1)
    c0 = (r * r).sum(-1) - R * R
    inside_start = c0 <= 0.0
    disc = b * b - 4.0 * a * c0
    has_real = disc >= 0.0
    sd = np.sqrt(np.maximum(disc, 0.0))
    a_safe = a + _EPS
    t1 = (-b - sd) / (2.0 * a_safe)
    t2 = (-b + sd) / (2.0 * a_safe)
    INF = np.full_like(t1, np.inf)
    t1_in = np.where((t1 >= 0.0) & (t1 <= 1.0), t1, INF)
    t2_in = np.where((t2 >= 0.0) & (t2 <= 1.0), t2, INF)
    t_min = np.minimum(t1_in, t2_in)
    t_min = np.where(has_real, t_min, INF)
    t_min = np.where(inside_start, 0.0, t_min)
    t_min = np.where(valid_mask, t_min, INF)
    t_min = np.where((a < _EPS) & (~inside_start), INF, t_min)
    return t_min


def segment_point_min_d2_np(p0, p1, q):
    v = p1 - p0
    rq = q - p0
    v2 = (v * v).sum(-1)
    t = (rq * v).sum(-1) / (v2 + _EPS)
    t = np.clip(t, 0.0, 1.0)
    closest = p0 + t[..., None] * v
    d = q - closest
    return (d * d).sum(-1)


@dataclass
class NpState:
    step: np.ndarray
    alive: np.ndarray
    angular_velocity: np.ndarray
    n_players: int
    planet_active: np.ndarray
    planet_is_comet: np.ndarray
    planet_orbiting: np.ndarray
    planet_owner: np.ndarray
    planet_pos: np.ndarray
    planet_radius: np.ndarray
    planet_ships: np.ndarray
    planet_production: np.ndarray
    planet_theta0: np.ndarray
    planet_r_sun: np.ndarray
    planet_path: np.ndarray
    planet_path_idx: np.ndarray
    planet_path_len: np.ndarray
    fleet_active: np.ndarray
    fleet_owner: np.ndarray
    fleet_pos: np.ndarray
    fleet_angle: np.ndarray
    fleet_ships: np.ndarray
    fleet_from: np.ndarray
    comet_spawn_step: np.ndarray
    comet_spawn_used: np.ndarray
    comet_spawn_paths: np.ndarray
    comet_spawn_path_len: np.ndarray
    comet_spawn_ships: np.ndarray
    max_steps: int = DEFAULT_EPISODE_STEPS
    just_spawned: np.ndarray = None  # (B,P) bool; comets spawned this transition

    @property
    def B(self): return int(self.step.shape[0])
    @property
    def P(self): return int(self.planet_active.shape[1])
    @property
    def F(self): return int(self.fleet_active.shape[1])


def empty_state_np(B, P, F, L, G=len(COMET_SPAWN_STEPS), n_players=2,
                   max_steps=DEFAULT_EPISODE_STEPS):
    return NpState(
        step=np.zeros(B, dtype=np.int64),
        alive=np.ones(B, dtype=bool),
        angular_velocity=np.full(B, 0.03, dtype=np.float32),
        n_players=n_players,
        planet_active=np.zeros((B, P), dtype=bool),
        planet_is_comet=np.zeros((B, P), dtype=bool),
        planet_orbiting=np.zeros((B, P), dtype=bool),
        planet_owner=np.full((B, P), NEUTRAL, dtype=np.int64),
        planet_pos=np.zeros((B, P, 2), dtype=np.float32),
        planet_radius=np.zeros((B, P), dtype=np.float32),
        planet_ships=np.zeros((B, P), dtype=np.int64),
        planet_production=np.zeros((B, P), dtype=np.int64),
        planet_theta0=np.zeros((B, P), dtype=np.float32),
        planet_r_sun=np.zeros((B, P), dtype=np.float32),
        planet_path=np.zeros((B, P, L, 2), dtype=np.float32),
        planet_path_idx=np.zeros((B, P), dtype=np.int64),
        planet_path_len=np.zeros((B, P), dtype=np.int64),
        fleet_active=np.zeros((B, F), dtype=bool),
        fleet_owner=np.zeros((B, F), dtype=np.int64),
        fleet_pos=np.zeros((B, F, 2), dtype=np.float32),
        fleet_angle=np.zeros((B, F), dtype=np.float32),
        fleet_ships=np.zeros((B, F), dtype=np.int64),
        fleet_from=np.zeros((B, F), dtype=np.int64),
        comet_spawn_step=np.full((B, G), -1, dtype=np.int64),
        comet_spawn_used=np.zeros((B, G), dtype=bool),
        comet_spawn_paths=np.zeros((B, G, 4, L, 2), dtype=np.float32),
        comet_spawn_path_len=np.zeros((B, G, 4), dtype=np.int64),
        comet_spawn_ships=np.zeros((B, G), dtype=np.int64),
        max_steps=max_steps,
        just_spawned=np.zeros((B, P), dtype=bool),
    )


class OrbitWarsNumpy:
    def __init__(self, max_steps=DEFAULT_EPISODE_STEPS):
        self.max_steps = max_steps

    def step(self, s, actions):
        # Turn order mirrors orbit_wars_reference.py:
        #   comet spawn -> fleet launch -> production -> planet/comet move
        #   -> fleet movement/collision -> combat -> comet expire -> step++.
        s = self._comet_spawn(s)
        s = self._fleet_launch(s, actions)
        s = self._production(s)
        s, old_pos = self._planet_move(s)
        s, planet_hits = self._fleet_move(s, old_pos)
        s = self._combat(s, planet_hits)
        s = self._comet_expire(s)
        s.step = s.step + 1
        s = self._update_alive(s)
        return s

    def _comet_expire(self, s):
        expired = s.planet_is_comet & (s.planet_path_idx >= s.planet_path_len)
        s.planet_active = s.planet_active & ~expired
        return s

    def _comet_spawn(self, s):
        B, P = s.B, s.P
        G = s.comet_spawn_step.shape[1]
        # Clear last step's just_spawned mask
        s.just_spawned = np.zeros((B, P), dtype=bool)
        step_b = s.step[:, None]
        # FIX: spawn during transition that produces state at spawn_step
        ready = (~s.comet_spawn_used) & (s.comet_spawn_step == step_b + 1)
        if not ready.any():
            return s
        for g in range(G):
            ready_g = ready[:, g]
            if not ready_g.any():
                continue
            inactive = ~s.planet_active
            inactive_f = inactive.astype(np.float32)
            order = np.argsort(-inactive_f, axis=1, kind="stable")
            target_slots = order[:, :4]
            for k in range(4):
                slot = target_slots[:, k]
                slot_onehot = np.zeros((B, P), dtype=np.float32)
                slot_onehot[np.arange(B), slot] = 1.0
                slot_mask = (slot_onehot > 0.5) & ready_g[:, None]
                slot_mask3 = slot_mask[..., None]
                s.planet_active = s.planet_active | slot_mask
                s.planet_is_comet = s.planet_is_comet | slot_mask
                s.just_spawned = s.just_spawned | slot_mask
                s.planet_orbiting = s.planet_orbiting & ~slot_mask
                s.planet_owner = np.where(slot_mask, NEUTRAL, s.planet_owner)
                init_pos = s.comet_spawn_paths[:, g, k, 0, :]
                pos_value = init_pos[:, None, :] * slot_onehot[..., None]
                s.planet_pos = np.where(slot_mask3, pos_value, s.planet_pos)
                s.planet_radius = np.where(slot_mask, 1.0, s.planet_radius)
                ships_val = s.comet_spawn_ships[:, g]
                s.planet_ships = np.where(slot_mask, ships_val[:, None], s.planet_ships)
                s.planet_production = np.where(slot_mask, 1, s.planet_production)
                s.planet_theta0 = np.where(slot_mask, 0.0, s.planet_theta0)
                s.planet_r_sun = np.where(slot_mask, 0.0, s.planet_r_sun)
                path_value = s.comet_spawn_paths[:, g, k, :, :]
                slot_mask_path = (slot_onehot[..., None, None].astype(bool)) & ready_g[:, None, None, None]
                s.planet_path = np.where(slot_mask_path, path_value[:, None, :, :], s.planet_path)
                s.planet_path_idx = np.where(slot_mask, 0, s.planet_path_idx)
                pl_len = s.comet_spawn_path_len[:, g, k]
                s.planet_path_len = np.where(slot_mask, pl_len[:, None], s.planet_path_len)
            s.comet_spawn_used[:, g] = s.comet_spawn_used[:, g] | ready_g
        return s

    def _fleet_launch(self, s, actions):
        """
        Launch fleets.

        Supports two action formats:

        1. Policy/vectorized format:
             actions[player_id] = (angle[B, P], ships[B, P])
           This allows at most one launch per source planet.

        2. Ordered replay-validation format:
             actions[player_id] = [[src_slot, angle, requested_ships], ...]
           This preserves duplicate same-source replay orders and applies them
           sequentially, matching orbit_wars_reference.py.
        """
        if self._actions_are_ordered_replay_lists(actions):
            return self._fleet_launch_ordered_replay(s, actions)

        B, P, F = s.B, s.P, s.F
        for player_id, (angle, ships) in actions.items():
            owned = s.planet_active & (s.planet_owner == player_id)
            ship_cap = np.maximum(ships, 0).astype(np.int64)
            ship_cap = np.minimum(ship_cap, s.planet_ships)
            launch = owned & (ship_cap > 0)
            deduction = np.where(launch, ship_cap, 0)
            s.planet_ships = s.planet_ships - deduction

            # spawn 0.1 units past the planet boundary (verified against engine)
            spawn_r = s.planet_radius + 0.1
            fx = s.planet_pos[..., 0] + spawn_r * np.cos(angle)
            fy = s.planet_pos[..., 1] + spawn_r * np.sin(angle)
            new_pos = np.stack([fx, fy], axis=-1)

            n_launch = int(launch.sum(-1).max()) if launch.size else 0
            if n_launch == 0:
                continue

            launch_f = launch.astype(np.float32)
            order = np.argsort(-launch_f, axis=1, kind="stable")
            top_slots = order[:, :n_launch]
            ra = np.arange(B)[:, None]

            top_mask = launch[ra, top_slots]
            top_angle = angle[ra, top_slots]
            top_ships = ship_cap[ra, top_slots]
            top_pos = new_pos[ra, top_slots]

            inactive_f = (~s.fleet_active).astype(np.float32)
            fleet_order = np.argsort(-inactive_f, axis=1, kind="stable")
            target_fslot = fleet_order[:, :n_launch]

            cur_active = s.fleet_active[ra, target_fslot]
            s.fleet_active[ra, target_fslot] = np.where(top_mask, True, cur_active)

            self._scatter(
                s.fleet_owner,
                ra,
                target_fslot,
                np.full_like(top_ships, player_id),
                top_mask,
            )
            self._scatter(
                s.fleet_angle,
                ra,
                target_fslot,
                top_angle.astype(s.fleet_angle.dtype),
                top_mask,
            )
            self._scatter(s.fleet_ships, ra, target_fslot, top_ships, top_mask)
            self._scatter(
                s.fleet_from,
                ra,
                target_fslot,
                top_slots.astype(np.int64),
                top_mask,
            )

            cur_pos = s.fleet_pos[ra, target_fslot]
            s.fleet_pos[ra, target_fslot] = np.where(
                top_mask[..., None],
                top_pos.astype(cur_pos.dtype),
                cur_pos,
            )

        return s

    @staticmethod
    def _actions_are_ordered_replay_lists(actions):
        """
        True for validation-time ordered action lists:
            actions[player_id] = [[src_slot, angle, ships], ...]

        False for normal policy arrays:
            actions[player_id] = (angle_array, ship_array)
        """
        if not isinstance(actions, dict):
            return False

        for value in actions.values():
            if isinstance(value, tuple) and len(value) == 2:
                return False

        return True

    def _fleet_launch_ordered_replay(self, s, actions):
        """
        Sequential replay launch path for B=1 validation.

        This mirrors orbit_wars_reference.py launch semantics, except source
        IDs have already been converted to source slots by the validator.
        """
        B, P, F = s.B, s.P, s.F
        if B != 1:
            raise ValueError(
                "ordered replay action lists are only supported for B=1 validation"
            )

        for player_id, orders in actions.items():
            for order in orders or []:
                if len(order) < 3:
                    continue

                src_slot = int(order[0])
                angle = float(order[1])
                requested_ships = int(order[2])

                if src_slot < 0 or src_slot >= P:
                    continue

                if not bool(s.planet_active[0, src_slot]):
                    continue

                if int(s.planet_owner[0, src_slot]) != int(player_id):
                    continue

                ships_before = int(s.planet_ships[0, src_slot])
                launch_ships = max(0, min(requested_ships, ships_before))
                if launch_ships <= 0:
                    continue

                inactive = np.where(~s.fleet_active[0])[0]
                if inactive.size <= 0:
                    active_count = int(np.sum(s.fleet_active[0]))
                    raise RuntimeError(
                        "No free fleet slots for ordered replay launch "
                        f"(F={F}, active={active_count}, "
                        f"player={player_id}, src_slot={src_slot}, "
                        f"requested_ships={requested_ships})"
                    )

                fslot = int(inactive[0])

                # Deduct immediately, then later orders from the same source
                # see the reduced garrison, exactly like the reference sim.
                s.planet_ships[0, src_slot] -= launch_ships

                spawn_r = float(s.planet_radius[0, src_slot]) + 0.1
                fx = float(s.planet_pos[0, src_slot, 0]) + spawn_r * math.cos(angle)
                fy = float(s.planet_pos[0, src_slot, 1]) + spawn_r * math.sin(angle)

                s.fleet_active[0, fslot] = True
                s.fleet_owner[0, fslot] = int(player_id)
                s.fleet_angle[0, fslot] = float(angle)
                s.fleet_ships[0, fslot] = int(launch_ships)
                s.fleet_from[0, fslot] = int(src_slot)
                s.fleet_pos[0, fslot, 0] = float(fx)
                s.fleet_pos[0, fslot, 1] = float(fy)

        return s
    
    @staticmethod
    def _scatter(dst, ra, idx, src, mask):
        cur = dst[ra, idx]
        dst[ra, idx] = np.where(mask, src.astype(dst.dtype), cur)

    def _production(self, s):
        owned = s.planet_active & (s.planet_owner >= 0)
        s.planet_ships = s.planet_ships + np.where(owned, s.planet_production, 0)
        return s

    def _fleet_move(self, s, planet_old_pos):
        """Fleet movement/collision matching orbit_wars_reference.py.

        Reference semantics:
          - planets/comets have already been moved, but collision uses
            old_pos -> new_pos as a moving disc trajectory;
          - just-spawned comets are skipped for this transition;
          - source planets are NOT skipped;
          - the first colliding planet in planet-slot iteration order wins,
            not the planet with the earliest collision time;
          - the sun kills the fleet if t_sun <= selected planet hit time;
          - OOB kills only after no sun/planet collision.
        """
        B, P, F = s.B, s.P, s.F

        spd = fleet_speed_np(s.fleet_ships)
        dx = spd * np.cos(s.fleet_angle)
        dy = spd * np.sin(s.fleet_angle)

        p0 = s.fleet_pos
        p1 = np.stack([p0[..., 0] + dx, p0[..., 1] + dy], axis=-1)

        active = s.fleet_active

        sun_c = np.broadcast_to(
            np.array([SUN_X, SUN_Y], dtype=np.float32),
            (B, F, 2),
        )
        sun_R = np.full((B, F), SUN_RADIUS, dtype=np.float32)
        t_sun = segment_enters_circle_np(p0, p1, sun_c, sun_R, active)

        p0e = np.broadcast_to(p0[:, :, None, :], (B, F, P, 2))
        p1e = np.broadcast_to(p1[:, :, None, :], (B, F, P, 2))
        c0e = np.broadcast_to(planet_old_pos[:, None, :, :], (B, F, P, 2))
        c1e = np.broadcast_to(s.planet_pos[:, None, :, :], (B, F, P, 2))
        pR = np.broadcast_to(s.planet_radius[:, None, :], (B, F, P))

        visible_planet = s.planet_active & ~s.just_spawned
        active_pp = active[:, :, None] & visible_planet[:, None, :]

        # Reference uses one relative-velocity moving-disc test for every
        # visible body. Static bodies naturally have c0 == c1.
        t_p = segment_enters_moving_circle_np(p0e, p1e, c0e, c1e, pR, active_pp)

        # Reference target selection is planet iteration order, not earliest t.
        hit_candidate = np.isfinite(t_p) & active_pp
        any_planet_hit = hit_candidate.any(axis=-1)
        first_idx = np.argmax(hit_candidate, axis=-1).astype(np.int64)
        first_t_raw = np.take_along_axis(t_p, first_idx[..., None], axis=-1).squeeze(-1)
        first_t = np.where(any_planet_hit, first_t_raw, np.inf)

        any_sun = np.isfinite(t_sun)
        dies_sun = any_sun & (t_sun <= first_t)

        oob = (p1[..., 0] < 0.0) | (p1[..., 0] > BOARD_SIZE) | \
              (p1[..., 1] < 0.0) | (p1[..., 1] > BOARD_SIZE)

        hit_planet = active & any_planet_hit & ~dies_sun
        planet_hit = np.where(hit_planet, first_idx, -1).astype(np.int64)

        survives = active & ~dies_sun & ~hit_planet & ~oob
        s.fleet_pos = np.where(survives[..., None], p1, p0)
        s.fleet_active = survives

        return s, planet_hit

    def _planet_move(self, s):
        """Move planets to new positions. Returns (s, old_pos) so fleet_move
        can use both for moving-disc collision."""
        B, P, F = s.B, s.P, s.F
        old_pos = s.planet_pos.copy()
        next_step = (s.step + 1)[:, None].astype(np.float32)
        eff_t = np.maximum(next_step - 1.0, 0.0)
        theta = s.planet_theta0 + eff_t * s.angular_velocity[:, None]
        orb_x = SUN_X + s.planet_r_sun * np.cos(theta)
        orb_y = SUN_Y + s.planet_r_sun * np.sin(theta)
        orb_pos = np.stack([orb_x, orb_y], axis=-1)
        next_idx = s.planet_path_idx + 1
        L = s.planet_path.shape[2]
        clamped_idx = np.minimum(next_idx, L - 1)
        bb = np.arange(B)[:, None]
        pp = np.arange(P)[None, :]
        comet_pos = s.planet_path[bb, pp, clamped_idx]
        out_of_path = next_idx >= s.planet_path_len
        comet_pos = np.where(out_of_path[..., None], s.planet_pos, comet_pos)
        # For comets: just-spawned stays at path[0], others advance.
        # Where just_spawned is True, comet position stays at current (path[0])
        # and path_idx stays at 0; otherwise the precomputed comet_pos applies.
        comet_pos_eff = np.where(s.just_spawned[..., None], s.planet_pos, comet_pos)
        new_pos = np.where(s.planet_is_comet[..., None], comet_pos_eff,
                           np.where(s.planet_orbiting[..., None], orb_pos, s.planet_pos))
        s.planet_pos = new_pos
        # Only advance path_idx for non-just-spawned active comets
        advance_comet = s.planet_is_comet & s.planet_active & ~s.just_spawned
        s.planet_path_idx = np.where(advance_comet, next_idx, s.planet_path_idx)
        return s, old_pos

    def _combat(self, s, planet_hits):
        B, P, F = s.B, s.P, s.F
        N = max(s.n_players, 4)
        arrivals = np.zeros((B, P, N), dtype=np.int64)

        valid_hit = planet_hits >= 0
        if valid_hit.any():
            owner = np.maximum(s.fleet_owner, 0)
            ships = np.where(valid_hit, s.fleet_ships, 0)
            flat_idx = np.maximum(planet_hits, 0) * N + owner
            flat = arrivals.reshape(B, P * N)
            for b in range(B):
                np.add.at(flat[b], flat_idx[b], ships[b])
            arrivals = flat.reshape(B, P, N)
        top_idx = np.argsort(-arrivals, axis=-1, kind="stable")[..., :2]
        top_ships = np.take_along_axis(arrivals, top_idx, axis=-1)
        attacker_force = top_ships[..., 0] - top_ships[..., 1]
        attacker_owner = top_idx[..., 0]
        tied = (attacker_force == 0) & (top_ships[..., 0] > 0)
        surviving_force = np.where(tied, 0, attacker_force)
        same_owner = (attacker_owner == s.planet_owner) & (surviving_force > 0)
        new_ships = np.where(same_owner, s.planet_ships + surviving_force, s.planet_ships)
        attacking = (~same_owner) & (surviving_force > 0)
        capture = attacking & (surviving_force > s.planet_ships)
        defends = attacking & ~capture
        new_owner = np.where(capture, attacker_owner, s.planet_owner)
        new_ships = np.where(capture, surviving_force - s.planet_ships, new_ships)
        new_ships = np.where(defends, np.maximum(s.planet_ships - surviving_force, 0), new_ships)
        new_owner = np.where(s.planet_active, new_owner, s.planet_owner)
        new_ships = np.where(s.planet_active, new_ships, s.planet_ships)
        s.planet_owner = new_owner
        s.planet_ships = new_ships
        return s

    def _update_alive(self, s):
        step_over = s.step >= s.max_steps
        ow = np.arange(s.n_players)
        owners_planets = (s.planet_owner[..., None] == ow[None, None, :]) & s.planet_active[..., None]
        has_planets = owners_planets.any(axis=1)
        owners_fleets = (s.fleet_owner[..., None] == ow[None, None, :]) & s.fleet_active[..., None]
        has_fleets = owners_fleets.any(axis=1)
        has_any = has_planets | has_fleets
        alive_count = has_any.sum(-1)
        elimination = alive_count <= 1
        s.alive = s.alive & ~(step_over | elimination)
        return s
 