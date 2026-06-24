"""
Observation Assembler for the Kaggle Orbit Wars competition.

Produces per-step token tensors for a transformer policy with three sets of
inputs:
    - planet/comet tokens : padded to MAX_PLANETS, both kinds share an encoder
    - fleet tokens        : padded to MAX_FLEETS
    - global features     : a small dense vector (turn, ship totals, etc.)

The assembler holds per-episode state (player perspective, orbital cache).
Call `reset(obs0, player_id)` once at the start of each episode, then
`assemble(obs, step)` each turn.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from orbit_obs_utils_reftrace import (
    BOARD_SIZE,
    COMET_SPAWN_STEPS,
    FLEET_VEL_HIGH,
    FLEET_VEL_LOW,
    LOG_SHIPS_HIGH,
    LOG_SHIPS_LOW,
    MAX_FLEETS,
    MAX_PLANETS,
    MAX_PLAYERS,
    N_ACTION_OPTIONS,
    N_BINS,
    N_SHIP_BINS,
    OPT_SKIP,
    OPT_MAX_SEND,
    PLANET_VEL_HIGH,
    PLANET_VEL_LOW,
    POS_HIGH,
    POS_LOW,
    SUN_X,
    SUN_Y,
    SUN_RADIUS,
    fleet_speed,
    is_orbiting,
    orbit_position,
    segment_enters_circle,
)
from orbit_solvers import lead_intercept, lead_intercept_path
from orbit_solvers_torch import batched_planet_intercepts

# Use the same vectorized collision primitives as the replay-validated
# NumPy simulator. This keeps incoming projection semantics tied to the sim
# oracle instead of maintaining a second independent collision kernel here.
from orbit_wars_numpy import (
    fleet_speed_np,
    segment_enters_circle_np,
    segment_enters_moving_circle_np,
)

# ---------------------------------------------------------------------------
# Token feature layouts
# ---------------------------------------------------------------------------
TTA_BUCKETS = 25  # Extended to 25 steps

# We compress the representation heavily here to save Replay Buffer RAM.
# The actual 51-bin two-hot expansions happen dynamically on the GPU.
PLANET_DIM = (
    4                                      # spatial
    + 1                                    # raw ln(ships)
    + 5                                    # prod
    + 5                                    # owner
    + 5                                    # flags
    + 1                                    # despawn
    + (4 * TTA_BUCKETS)                    # raw incoming 4-player matrix ln(ships)
    + TTA_BUCKETS                          # projected garrison ln(ships)
    + (5 * TTA_BUCKETS)                    # projected owner slot
)

GLOBAL_DIM = 31

# Planet Offsets
_P_OFF_X = 0
_P_OFF_Y = 1
_P_OFF_DX = 2
_P_OFF_DY = 3
_P_OFF_SHIPS = 4
_P_OFF_PROD = 5
_P_OFF_OWNER = 10
_P_OFF_FLAGS = 15
_P_OFF_DESPAWN = 20
_P_OFF_INC = 21
_P_OFF_GARRISON = _P_OFF_INC + (4 * TTA_BUCKETS)
_P_OFF_PROJ_OWNER = _P_OFF_GARRISON + TTA_BUCKETS

# ---------------------------------------------------------------------------
# Directed source-target edge feature layout
# ---------------------------------------------------------------------------
# Binary action-space edge features:
#   0..49: max-send ETA one-hot bucket
#          bucket 0  = ETA <= 1 turn
#          bucket 49 = ETA >= 50 turns, blocked, or unknown
#   50: max-send blocked flag
#   51: can_takeover_with_max_send flag
#   52..56: source owner slot one-hot
#   57..61: target owner slot one-hot
EDGE_ETA_BUCKETS = 50
EDGE_TTA_MAX = float(EDGE_ETA_BUCKETS)
EDGE_BLOCKED_TTA_NORM = 1.25

_E_OFF_MAX_TTA = 0
_E_OFF_MAX_TTA_BUCKETS = _E_OFF_MAX_TTA
_E_OFF_MAX_BLOCKED = _E_OFF_MAX_TTA + EDGE_ETA_BUCKETS
_E_OFF_CAN_TAKEOVER = _E_OFF_MAX_BLOCKED + 1

_E_OFF_SRC_OWNER = _E_OFF_CAN_TAKEOVER + 1  # Length 5
_E_OFF_TGT_OWNER = _E_OFF_SRC_OWNER + 5     # Length 5

EDGE_DIM = _E_OFF_TGT_OWNER + 5

def _angle_delta(a: float, b: float) -> float:
    """Smallest absolute angular difference between two angles."""
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))

def _segment_hits_sun(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    *,
    sun_x: float = SUN_X,
    sun_y: float = SUN_Y,
    sun_radius: float = SUN_RADIUS,
    eps: float = 1e-9,
) -> bool:
    """
    True if the segment p0 -> p1 passes through the sun.
    """
    x0, y0 = p0
    x1, y1 = p1

    vx = x1 - x0
    vy = y1 - y0

    wx = sun_x - x0
    wy = sun_y - y0

    denom = vx * vx + vy * vy
    if denom <= eps:
        return math.hypot(x0 - sun_x, y0 - sun_y) <= sun_radius

    t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    cx = x0 + t * vx
    cy = y0 + t * vy

    return math.hypot(cx - sun_x, cy - sun_y) <= sun_radius

def _solve_with_launch_offset(
    src_center: Tuple[float, float],
    src_radius: float,
    solve_from_origin,
    *,
    launch_margin: float = 0.1,
    max_iter: int = 6,
    angle_tol: float = 1e-5,
):
    """
    Solves the fixed-point loop between launch point and intercept angle.
    Returns (angle, intercept_time) or None.
    """
    solved = solve_from_origin(src_center)
    if solved is None:
        return None

    angle, t_hit = solved

    for _ in range(max_iter):
        launch_point = (
            src_center[0] + math.cos(angle) * (src_radius + launch_margin),
            src_center[1] + math.sin(angle) * (src_radius + launch_margin),
        )

        solved = solve_from_origin(launch_point)
        if solved is None:
            return None

        new_angle, new_t_hit = solved

        if _angle_delta(new_angle, angle) < angle_tol:
            return new_angle, new_t_hit

        angle, t_hit = new_angle, new_t_hit

    return angle, t_hit

class OrbitWarsAssembler:
    """Per-agent observation builder. One instance per environment per agent."""

    def __init__(self, max_steps=500):
        self.max_steps = max_steps
        self.planet_buf = np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)
        self.planet_mask = np.zeros(MAX_PLANETS, dtype=np.float32)
        self.global_buf = np.zeros(GLOBAL_DIM, dtype=np.float32)
        self.source_mask = np.zeros(MAX_PLANETS, dtype=np.float32)
        
        self.edge_buf = np.zeros((MAX_PLANETS, MAX_PLANETS, EDGE_DIM), dtype=np.float32)
        self.edge_mask = np.zeros((MAX_PLANETS, MAX_PLANETS), dtype=np.float32)
        
        self.FAST_SPEED_SHIPS = 150
        self.fast_speed_const = fleet_speed(self.FAST_SPEED_SHIPS)
        self.slow_speed_const = fleet_speed(1)
        
        self._reset_episode_state()

    def _reset_episode_state(self):
        self.player_id = None
        self.angular_velocity = None
        self.r0 = {}
        self.theta_0 = {}
        self.is_orbiting_cache = {}
        self.home_planet_ids = set()
        self.owner_slot = {}
        self.planet_paths = {}
        
        # --- THE FLEET RADAR & TENSORIZED CACHE ---
        self.fleet_radar = {}  
        self.known_comets = set()
        self.static_cache_built = False
        self.target_reserved_until = {}
        
        self.cache_s_eta = np.full((MAX_PLANETS, MAX_PLANETS), np.nan, dtype=np.float32)
        self.cache_s_blk = np.ones((MAX_PLANETS, MAX_PLANETS), dtype=bool)
        self.cache_f_eta = np.full((MAX_PLANETS, MAX_PLANETS), np.nan, dtype=np.float32)
        self.cache_f_blk = np.ones((MAX_PLANETS, MAX_PLANETS), dtype=bool)

    def reset(self, obs, player_id):
        self._reset_episode_state()
        self.player_id = int(player_id)
        self.angular_velocity = float(obs.get("angular_velocity", 0.0))

        initial = obs.get("initial_planets") or obs.get("planets") or []
        for row in initial:
            pid, owner, x, y, radius, _ships, _prod = row
            dx0 = x - SUN_X
            dy0 = y - SUN_Y
            r = math.hypot(dx0, dy0)
            self.r0[pid] = r
            self.theta_0[pid] = math.atan2(dy0, dx0)
            orb = is_orbiting(r, radius)
            self.is_orbiting_cache[pid] = orb
            
            if owner == self.player_id:
                self.home_planet_ids.add(pid)

            if orb and self.angular_velocity != 0.0:
                # Dense [max_steps+1, 2] float32 path, vectorized once.
                # Equivalent to the per-step ``orbit_position`` calls but pays
                # the cost in NumPy rather than a Python loop, and lets the
                # tracer slice instead of indexing list-of-tuples.
                ts = np.arange(self.max_steps + 1, dtype=np.float32)
                eff_t = np.maximum(0.0, ts - 1.0)
                thetas = float(self.theta_0[pid]) + eff_t * float(self.angular_velocity)
                path = np.empty((self.max_steps + 1, 2), dtype=np.float32)
                path[:, 0] = SUN_X + r * np.cos(thetas)
                path[:, 1] = SUN_Y + r * np.sin(thetas)
                self.planet_paths[pid] = path

        opps = [pid for pid in range(MAX_PLAYERS) if pid != self.player_id]
        self.owner_slot = {self.player_id: 0, -1: 4}
        for slot, raw in enumerate(opps, start=1):
            self.owner_slot[raw] = slot
            
    def _compute_expert_action(self, planets, step, ally_inc, enemy_inc, comet_info):
        """
        Thin wrapper over ``heuristic_policy.compute_expert_action_vectorized``.

        Kept on the assembler so callers that hold an
        ``OrbitWarsAssembler`` keep a one-call entry point and so the
        per-episode ``target_reserved_until`` reservation map lives on
        the assembler (cleared by ``_reset_episode_state``). The actual
        scoring is the vectorized NumPy implementation in
        ``heuristic_policy.py``; the magic numbers live on
        ``ExpertActionConfig`` and can be swept via
        ``heuristics_sweep.py``.
        """
        from expert_action import (
            ExpertActionConfig,
            compute_expert_action_vectorized,
        )

        cfg = getattr(self, "_expert_cfg", None)
        if cfg is None:
            cfg = ExpertActionConfig()
            self._expert_cfg = cfg

        rng = getattr(self, "_expert_rng", None)
        if rng is None:
            rng = np.random.default_rng()
            self._expert_rng = rng

        n_p = min(len(planets), MAX_PLANETS)
        return compute_expert_action_vectorized(
            planets=planets,
            n_p=n_p,
            step=int(step),
            ally_inc=ally_inc,
            enemy_inc=enemy_inc,
            edge_max_tta=self._eta_bucket_features_to_norm(),
            edge_max_blocked=self.edge_buf[..., _E_OFF_MAX_BLOCKED],
            edge_tta_max=EDGE_TTA_MAX,
            tta_buckets=TTA_BUCKETS,
            player_id=int(self.player_id),
            target_reserved_until=self.target_reserved_until,
            cfg=cfg,
            comet_info=comet_info,
            rng=rng,
        )

    def _body_position_at(self, planet, abs_step: int, comet_info):
        """Return body center at absolute engine step `abs_step`, or None if expired."""
        pid, _owner, x, y, _radius, _ships, _prod = planet
        pid = int(pid)

        if pid in comet_info:
            path, pidx, plen = comet_info[pid]
            rel_idx = int(pidx) + (int(abs_step) - int(getattr(self, "_trace_base_step", abs_step)))
            if rel_idx < 0 or rel_idx >= plen:
                return None
            return float(path[rel_idx][0]), float(path[rel_idx][1])

        if pid in self.planet_paths:
            idx = max(0, min(int(abs_step), len(self.planet_paths[pid]) - 1))
            return self.planet_paths[pid][idx]

        return float(x), float(y)

    def _body_transition(self, planet, base_step: int, offset: int, comet_info):
        """Return (old_x, old_y, new_x, new_y), or None if body is expired."""
        pid, _owner, x, y, _radius, _ships, _prod = planet
        pid = int(pid)
        abs_step = int(base_step) + int(offset)

        if pid in comet_info:
            path, pidx, plen = comet_info[pid]
            old_idx = int(pidx) + int(offset)
            if old_idx < 0 or old_idx >= plen:
                return None
            new_idx = old_idx + 1
            old = (float(path[old_idx][0]), float(path[old_idx][1]))
            if new_idx < plen:
                new = (float(path[new_idx][0]), float(path[new_idx][1]))
            else:
                new = old
            return old[0], old[1], new[0], new[1]

        if pid in self.planet_paths:
            path = self.planet_paths[pid]
            old_idx = max(0, min(abs_step, len(path) - 1))
            new_idx = max(0, min(abs_step + 1, len(path) - 1))
            old = path[old_idx]
            new = path[new_idx]
            return float(old[0]), float(old[1]), float(new[0]), float(new[1])

        px, py = float(x), float(y)
        return px, py, px, py

    def _body_transition_arrays(self, planets, base_step: int, offset: int, comet_info):
        """Vectorized body transitions for one projected tick.

        Returns active body ids, radii, old positions, and new positions in
        current planet-list order. This mirrors the body trajectory inputs used
        by the validated NumPy/reference fleet collision pass.
        """
        n_p = min(len(planets), MAX_PLANETS)
        if n_p <= 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=bool),
            )

        pids = np.zeros(n_p, dtype=np.int64)
        radii = np.zeros(n_p, dtype=np.float32)
        old_pos = np.zeros((n_p, 2), dtype=np.float32)
        new_pos = np.zeros((n_p, 2), dtype=np.float32)
        active = np.zeros(n_p, dtype=bool)

        for i, pl in enumerate(planets[:n_p]):
            pids[i] = int(pl[0])
            radii[i] = float(pl[4])
            trans = self._body_transition(pl, int(base_step), int(offset), comet_info)
            if trans is None:
                continue

            px0, py0, px1, py1 = trans
            old_pos[i, 0] = float(px0)
            old_pos[i, 1] = float(py0)
            new_pos[i, 0] = float(px1)
            new_pos[i, 1] = float(py1)
            active[i] = True

        return pids, radii, old_pos, new_pos, active

    def _compute_body_path_tensor(
        self,
        planets,
        base_step: int,
        horizon: int,
        comet_info,
    ):
        """Precompute body geometry across ``horizon + 1`` ticks in one pass.

        Returns ``(pids, radii, body_pos, body_active)`` where:
          - ``pids``        : [n_p] int64 body identifiers
          - ``radii``       : [n_p] float32 hit radii
          - ``body_pos``    : [T, n_p, 2] float32 position at each tick offset
          - ``body_active`` : [T, n_p] bool (False for expired comet bodies)

        ``T == horizon + 1``. Body kind is classified once per call (static /
        orbiting / comet) and positions are filled vectorized per kind, so the
        replay-validated tracer can slice ``body_pos[off]`` / ``body_pos[off+1]``
        instead of rebuilding the body grid in a Python loop every tick.
        """
        n_p = min(len(planets), MAX_PLANETS)
        T = int(horizon) + 1

        if n_p <= 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
                np.zeros((T, 0, 2), dtype=np.float32),
                np.zeros((T, 0), dtype=bool),
            )

        pids = np.zeros(n_p, dtype=np.int64)
        radii = np.zeros(n_p, dtype=np.float32)
        static_x = np.zeros(n_p, dtype=np.float32)
        static_y = np.zeros(n_p, dtype=np.float32)
        is_comet_mask = np.zeros(n_p, dtype=bool)
        is_orb_mask = np.zeros(n_p, dtype=bool)
        orb_r = np.zeros(n_p, dtype=np.float32)
        orb_theta0 = np.zeros(n_p, dtype=np.float32)

        comet_body_idx: List[int] = []
        comet_paths: List[np.ndarray] = []
        comet_pidx: List[int] = []

        planet_paths = self.planet_paths
        for i, pl in enumerate(planets[:n_p]):
            pid = int(pl[0])
            pids[i] = pid
            radii[i] = float(pl[4])
            static_x[i] = float(pl[2])
            static_y[i] = float(pl[3])

            if pid in comet_info:
                is_comet_mask[i] = True
                path, pidx, _plen = comet_info[pid]
                comet_body_idx.append(i)
                comet_paths.append(np.asarray(path, dtype=np.float32))
                comet_pidx.append(int(pidx))
            elif pid in planet_paths:
                is_orb_mask[i] = True
                orb_r[i] = float(self.r0[pid])
                orb_theta0[i] = float(self.theta_0[pid])

        body_pos = np.empty((T, n_p, 2), dtype=np.float32)
        body_active = np.ones((T, n_p), dtype=bool)

        # Static bodies (and orbiting planets when omega == 0, since
        # planet_paths is not built in that case).
        static_mask = ~is_comet_mask & ~is_orb_mask
        if static_mask.any():
            body_pos[:, static_mask, 0] = static_x[static_mask]
            body_pos[:, static_mask, 1] = static_y[static_mask]

        # Orbiting bodies: vectorized orbit_position over [T, M_orb].
        if is_orb_mask.any():
            orb_idx = np.where(is_orb_mask)[0]
            ks = np.arange(T, dtype=np.int64) + int(base_step)
            eff_t = np.maximum(0, ks - 1).astype(np.float32)
            omega = float(self.angular_velocity or 0.0)
            thetas = orb_theta0[orb_idx][None, :] + eff_t[:, None] * omega
            rr = orb_r[orb_idx][None, :]
            body_pos[:, orb_idx, 0] = SUN_X + rr * np.cos(thetas)
            body_pos[:, orb_idx, 1] = SUN_Y + rr * np.sin(thetas)

        # Comet bodies: per-body slice with out-of-range bodies deactivated.
        if comet_body_idx:
            offsets = np.arange(T, dtype=np.int64)
            for k, i in enumerate(comet_body_idx):
                path = comet_paths[k]
                plen = int(path.shape[0])
                pidx = comet_pidx[k]
                rel = pidx + offsets
                in_bounds = (rel >= 0) & (rel < plen)
                # ``rel`` is clamped to a valid index for the gather so we get
                # contiguous reads; ``body_active`` carries the validity bit.
                safe_rel = np.where(in_bounds, rel, 0)
                body_pos[:, i, :] = path[safe_rel]
                body_active[:, i] = in_bounds

        return pids, radii, body_pos, body_active

    def _trace_fleets_numpy_engine_semantics(
        self,
        fleets,
        *,
        planets,
        current_step: int,
        comet_info,
        max_turns: Optional[int] = None,
        speeds: Optional[np.ndarray] = None,
    ):
        """Batch-project fleets using the replay-validated NumPy collision semantics.

        Returns (target_pids, etas, blocked), one row per input fleet.
        target_pids is -1 for no planet hit. etas is continuous turns from the
        current observation; blocked=True means sun/OOB/no-hit-within-horizon.

        This is intentionally the same selection rule as orbit_wars_numpy.py:
          - one moving-disc segment test for every visible body;
          - first colliding body in planet-list/slot order wins;
          - sun kills if t_sun <= selected planet hit time;
          - OOB kills only after no sun/planet collision.
        """
        fleets = list(fleets or [])
        n_f = len(fleets)

        target_pids = np.full(n_f, -1, dtype=np.int64)
        etas = np.full(n_f, np.nan, dtype=np.float32)
        blocked = np.ones(n_f, dtype=bool)
        if n_f <= 0:
            return target_pids, etas, blocked

        if max_turns is None:
            max_turns = self.max_steps - int(current_step)
        max_turns = max(0, int(max_turns))

        pos = np.array([[float(f[2]), float(f[3])] for f in fleets], dtype=np.float32)
        angles = np.array([float(f[4]) for f in fleets], dtype=np.float32)

        if speeds is None:
            ships = np.array([int(f[6]) for f in fleets], dtype=np.int64)
            speed = fleet_speed_np(ships).astype(np.float32)
        else:
            speed = np.asarray(speeds, dtype=np.float32).reshape(n_f)

        active = speed > 0.0
        vx = speed * np.cos(angles)
        vy = speed * np.sin(angles)

        # Hoist invariants out of the off loop: body geometry, body radii,
        # sun centers/radii. The previous implementation rebuilt all of these
        # in Python every tick, which dominated the tracer cost.
        pids_body, radii_body, body_pos, body_active_t = self._compute_body_path_tensor(
            planets, int(current_step), max_turns, comet_info,
        )
        n_b = int(pids_body.shape[0])

        sun_c = np.broadcast_to(
            np.array([SUN_X, SUN_Y], dtype=np.float32),
            (n_f, 2),
        )
        sun_R = np.full((n_f,), SUN_RADIUS, dtype=np.float32)

        if n_b > 0:
            pR_full = np.broadcast_to(radii_body[None, :], (n_f, n_b))

        for off in range(max_turns):
            if not bool(active.any()):
                break

            next_pos = np.stack([pos[:, 0] + vx, pos[:, 1] + vy], axis=-1)

            t_sun = segment_enters_circle_np(pos, next_pos, sun_c, sun_R, active)

            if n_b > 0:
                old_body_pos = body_pos[off]
                new_body_pos = body_pos[off + 1]
                body_active = body_active_t[off] & body_active_t[off + 1]

                p0e = np.broadcast_to(pos[:, None, :], (n_f, n_b, 2))
                p1e = np.broadcast_to(next_pos[:, None, :], (n_f, n_b, 2))
                c0e = np.broadcast_to(old_body_pos[None, :, :], (n_f, n_b, 2))
                c1e = np.broadcast_to(new_body_pos[None, :, :], (n_f, n_b, 2))
                active_pp = active[:, None] & body_active[None, :]

                t_p = segment_enters_moving_circle_np(p0e, p1e, c0e, c1e, pR_full, active_pp)

                hit_candidate = np.isfinite(t_p) & active_pp
                any_planet_hit = hit_candidate.any(axis=-1)
                first_idx = np.argmax(hit_candidate, axis=-1).astype(np.int64)
                first_t_raw = np.take_along_axis(t_p, first_idx[:, None], axis=-1).squeeze(-1)
                first_t = np.where(any_planet_hit, first_t_raw, np.inf).astype(np.float32)
                first_pid = pids_body[first_idx]
            else:
                any_planet_hit = np.zeros(n_f, dtype=bool)
                first_t = np.full(n_f, np.inf, dtype=np.float32)
                first_pid = np.full(n_f, -1, dtype=np.int64)

            any_sun = np.isfinite(t_sun)
            dies_sun = active & any_sun & (t_sun <= first_t)

            oob = active & (
                (next_pos[:, 0] < 0.0)
                | (next_pos[:, 0] > BOARD_SIZE)
                | (next_pos[:, 1] < 0.0)
                | (next_pos[:, 1] > BOARD_SIZE)
            )

            hit_planet = active & any_planet_hit & ~dies_sun

            if bool(hit_planet.any()):
                target_pids[hit_planet] = first_pid[hit_planet]
                etas[hit_planet] = float(off) + first_t[hit_planet]
                blocked[hit_planet] = False

            if bool(dies_sun.any()):
                etas[dies_sun] = float(off) + t_sun[dies_sun].astype(np.float32)

            if bool(oob.any()):
                etas[oob] = float(off) + 1.0

            terminal = hit_planet | dies_sun | oob
            active = active & ~terminal
            pos[active] = next_pos[active]

        return target_pids, etas, blocked

    def _trace_fleet_engine_semantics(
        self,
        *,
        fx: float,
        fy: float,
        angle: float,
        ships: Optional[int] = None,
        speed: Optional[float] = None,
        planets,
        current_step: int,
        comet_info,
        max_turns: Optional[int] = None,
    ):
        """Reference-compatible forward trace for one fleet with no future actions.

        This scalar API is kept for edge validation and action decoding, but
        internally routes through the same replay-validated NumPy batch kernel
        used by incoming-fleet projection.
        """
        if speed is None:
            speed = fleet_speed(int(ships or 0))

        dummy_ships = int(ships or 1)
        fleet = [-1, -1, float(fx), float(fy), float(angle), -1, dummy_ships]
        speeds = np.array([float(speed)], dtype=np.float32)

        target_pids, etas, blocked = self._trace_fleets_numpy_engine_semantics(
            [fleet],
            planets=planets,
            current_step=int(current_step),
            comet_info=comet_info,
            max_turns=max_turns,
            speeds=speeds,
        )

        target_pid = int(target_pids[0])
        eta = None if not np.isfinite(etas[0]) else float(etas[0])
        return (None if target_pid < 0 else target_pid), eta, bool(blocked[0])

    def _raycast_fleet(self, fleet, obs, current_step, comet_info):
        """Project an existing fleet using the validated reference collision semantics."""
        _fid, _owner, fx, fy, angle, _from_pid, ships = fleet
        planets = (obs.get("planets") or [])[:MAX_PLANETS]
        target_pid, eta, blocked = self._trace_fleet_engine_semantics(
            fx=float(fx),
            fy=float(fy),
            angle=float(angle),
            ships=int(ships),
            planets=planets,
            current_step=int(current_step),
            comet_info=comet_info,
            max_turns=self.max_steps - int(current_step),
        )
        if blocked or target_pid is None or eta is None:
            return (None, -1)
        # A collision with eta in (0, 1] appears in obs_{step+1};
        # eta in (1, 2] appears in obs_{step+2}, etc.
        # The tiny epsilon keeps exact integer ETAs in the earlier frame,
        # matching transition semantics.
        eta_f = max(0.0, float(eta))
        arrival_delta = max(1, int(math.ceil(eta_f - 1e-9)))
        arrival_step = int(current_step) + arrival_delta
        return (target_pid, arrival_step)
    
    def _make_target_solver(self, target_planet, step, comet_info):
        tgt_pid, _tgt_owner, tgt_x, tgt_y, tgt_radius, _tgt_ships, _tgt_prod = target_planet
    
        tgt_pid = int(tgt_pid)
        tgt_x = float(tgt_x)
        tgt_y = float(tgt_y)
        tgt_radius = float(tgt_radius)
    
        if tgt_pid in comet_info:
            path, pidx, _plen = comet_info[tgt_pid]
            def make_solver(speed):
                def solve_from_origin(origin):
                    return lead_intercept_path(origin, path, pidx, speed, hit_radius=tgt_radius)
                return solve_from_origin
            return make_solver
    
        if tgt_pid in self.planet_paths:
            path = self.planet_paths[tgt_pid]
            def make_solver(speed):
                def solve_from_origin(origin):
                    return lead_intercept_path(origin, path, step, speed, hit_radius=tgt_radius)
                return solve_from_origin
            return make_solver
    
        def make_solver(speed):
            def solve_from_origin(origin):
                return lead_intercept(origin, (tgt_x, tgt_y), (0.0, 0.0), speed)
            return solve_from_origin
    
        return make_solver
    
    def _solve_edge_eta(
        self,
        src_planet,
        tgt_planet,
        speed: float,
        step: int,
        comet_info,
        planets=None,
    ) -> Tuple[Optional[float], bool]:
        """
        Propose an intercept angle, then validate the actual target/ETA with
        the reference-compatible fleet trace. Returns (eta, blocked).
        """
        if speed <= 0.0:
            return None, True

        if planets is None:
            planets = [src_planet, tgt_planet]

        _src_pid, _src_owner, src_x, src_y, src_radius, _src_ships, _src_prod = src_planet
        tgt_pid = int(tgt_planet[0])
        src_center = (float(src_x), float(src_y))
        src_radius = float(src_radius)

        make_solver = self._make_target_solver(tgt_planet, step, comet_info)

        solved = _solve_with_launch_offset(
            src_center,
            src_radius,
            make_solver(speed),
        )

        if solved is None:
            return None, True

        angle, _t_hit = solved
        if not math.isfinite(float(angle)):
            return None, True

        launch_point = (
            src_center[0] + math.cos(angle) * (src_radius + 0.1),
            src_center[1] + math.sin(angle) * (src_radius + 0.1),
        )

        # Edge ETAs only matter up to ``EDGE_TTA_MAX`` (the policy edge feature
        # caps there too), so a small horizon is sufficient and avoids paying
        # the full ``max_steps - step`` projection cost per comet-target pair.
        remaining = self.max_steps - int(step)
        edge_horizon = max(1, min(remaining, int(math.ceil(EDGE_TTA_MAX)) + 2))
        hit_pid, eta, blocked = self._trace_fleet_engine_semantics(
            fx=launch_point[0],
            fy=launch_point[1],
            angle=float(angle),
            speed=float(speed),
            planets=planets,
            current_step=int(step),
            comet_info=comet_info,
            max_turns=edge_horizon,
        )

        if blocked or hit_pid != tgt_pid or eta is None:
            return None, True

        return float(eta), False

    def _scan_angle_to_target_exact(
        self,
        *,
        src_center: Tuple[float, float],
        src_radius: float,
        ship_count: int,
        tgt_pid: int,
        planets,
        current_step: int,
        comet_info,
        max_turns: Optional[int] = None,
        seed_angle: Optional[float] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Exact fallback for map_actions_to_orders().

        The analytic intercept solver can fail even when a valid engine angle
        exists. This fallback generates candidate launch angles, then validates
        all candidates in one batch through the replay-validated NumPy trace.

        Returns (angle, eta) for the earliest candidate that actually hits
        tgt_pid, or None.
        """
        ship_count = int(ship_count)
        if ship_count <= 0:
            return None

        if max_turns is None:
            max_turns = self.max_steps - int(current_step)
        max_turns = max(1, int(max_turns))

        speed = float(fleet_speed(ship_count))
        if speed <= 0.0:
            return None

        tgt_pid = int(tgt_pid)
        tgt_planet = None
        for pl in planets:
            if int(pl[0]) == tgt_pid:
                tgt_planet = pl
                break

        if tgt_planet is None:
            return None

        sx, sy = float(src_center[0]), float(src_center[1])
        sr = float(src_radius)
        tgt_radius = float(tgt_planet[4])

        candidates: List[float] = []

        def add_angle(a: float):
            if math.isfinite(float(a)):
                candidates.append(float(math.atan2(math.sin(a), math.cos(a))))

        # 1. Try the analytic angle neighborhood first when available.
        if seed_angle is not None and math.isfinite(float(seed_angle)):
            seed = float(seed_angle)
            for delta in np.linspace(-0.25, 0.25, 81):
                add_angle(seed + float(delta))

        # 2. Target-path guided candidates. This is usually enough and avoids
        # a blind full-circle search for moving/orbiting targets.
        guided_horizon = min(max_turns, max(1, int(math.ceil(EDGE_TTA_MAX)) + 5))
        for off in range(guided_horizon + 1):
            trans = self._body_transition(tgt_planet, int(current_step), int(off), comet_info)
            if trans is None:
                break

            # Add bearings to both the old and new body positions for that tick.
            for tx, ty in ((trans[0], trans[1]), (trans[2], trans[3])):
                dx = float(tx) - sx
                dy = float(ty) - sy
                dist = math.hypot(dx, dy)
                if dist <= 1e-6:
                    continue

                center = math.atan2(dy, dx)
                width = math.asin(min(0.95, (tgt_radius + sr + 0.25) / max(dist, 1e-6)))
                width = max(width, 0.01)

                for mul in (0.0, -1.0, 1.0, -0.5, 0.5, -1.5, 1.5):
                    add_angle(center + mul * width)

        # 3. Coarse full-circle safety net. This only runs inside the fallback,
        # not for normal action decoding.
        for a in np.linspace(-math.pi, math.pi, 1440, endpoint=False):
            add_angle(float(a))

        if not candidates:
            return None

        angles = np.asarray(candidates, dtype=np.float32)
        angles = np.unique(np.round(angles, 6)).astype(np.float32)
        n = int(len(angles))
        if n <= 0:
            return None

        launch_r = sr + 0.1
        lx = sx + np.cos(angles) * launch_r
        ly = sy + np.sin(angles) * launch_r

        fleets = [
            [
                -1,
                -1,
                float(lx[k]),
                float(ly[k]),
                float(angles[k]),
                -1,
                ship_count,
            ]
            for k in range(n)
        ]

        speeds = np.full(n, speed, dtype=np.float32)

        hit_pids, etas, blocked = self._trace_fleets_numpy_engine_semantics(
            fleets,
            planets=planets,
            current_step=int(current_step),
            comet_info=comet_info,
            max_turns=max_turns,
            speeds=speeds,
        )

        good = (
            (~blocked)
            & np.isfinite(etas)
            & (hit_pids.astype(np.int64) == int(tgt_pid))
        )

        if not bool(np.any(good)):
            return None

        good_idx = np.where(good)[0]
        best_local = int(good_idx[np.argmin(etas[good_idx])])
        return float(angles[best_local]), float(etas[best_local])

    def _eta_to_bucket(self, eta: Optional[float], blocked: bool) -> int:
        """
        Convert max-send ETA to a 50-bucket categorical feature.

        bucket 0  = ETA <= 1
        bucket 49 = ETA >= 50, blocked, or unknown
        """
        if blocked or eta is None or not math.isfinite(float(eta)):
            return EDGE_ETA_BUCKETS - 1

        # ceil(eta) maps (0, 1] -> 1, then subtract 1 for zero-index bucket.
        b = int(math.ceil(max(0.0, float(eta)))) - 1
        return max(0, min(b, EDGE_ETA_BUCKETS - 1))

    def _eta_onehot_matrix(self, etas: np.ndarray, blocked: np.ndarray) -> np.ndarray:
        """
        Vectorized ETA one-hot encoding for directed edge features.
        """
        n = int(len(etas))
        out = np.zeros((n, EDGE_ETA_BUCKETS), dtype=np.float32)
        if n <= 0:
            return out

        finite = np.isfinite(etas)
        buckets = np.full(n, EDGE_ETA_BUCKETS - 1, dtype=np.int64)

        valid = (~blocked) & finite
        if np.any(valid):
            b = np.ceil(np.maximum(etas[valid], 0.0)).astype(np.int64) - 1
            buckets[valid] = np.clip(b, 0, EDGE_ETA_BUCKETS - 1)

        out[np.arange(n), buckets] = 1.0
        return out

    def _eta_bucket_features_to_norm(self) -> np.ndarray:
        """
        Reconstruct an approximate normalized ETA scalar from the one-hot edge
        buckets for legacy expert scoring.

        Returns a [MAX_PLANETS, MAX_PLANETS] matrix in [0, 1], where 1.0 means
        bucket 50 / clipped ETA.
        """
        eta_oh = self.edge_buf[
            ...,
            _E_OFF_MAX_TTA_BUCKETS : _E_OFF_MAX_TTA_BUCKETS + EDGE_ETA_BUCKETS,
        ]
        bucket_idx = np.argmax(eta_oh, axis=-1).astype(np.float32)
        return (bucket_idx + 1.0) / float(EDGE_ETA_BUCKETS)
    
    def _build_garrison_timeline(self, planets, inc_matrix):
        """Builds exact multi-way combat simulation over TTA_BUCKETS."""
        n_p = min(len(planets), MAX_PLANETS)
        timeline_ships = np.zeros((n_p, TTA_BUCKETS + 1), dtype=np.float32)
        timeline_slots = np.zeros((n_p, TTA_BUCKETS + 1), dtype=np.int32)
        prod = np.zeros(n_p, dtype=np.float32)

        defenders = np.zeros(n_p, dtype=np.float32)
        own_slot = np.zeros(n_p, dtype=np.int32)

        for i, p in enumerate(planets[:n_p]):
            pid, owner, x, y, r, ships, p_prod = p
            defenders[i] = max(int(ships), 0)
            own_slot[i] = self.owner_slot.get(int(owner), 4)
            prod[i] = float(p_prod)

        timeline_ships[:, 0] = defenders
        timeline_slots[:, 0] = own_slot

        for b in range(TTA_BUCKETS):
            # 1. Accrue production for owned planets
            defenders += np.where(own_slot != 4, prod, 0.0)

            # 2. Extract arriving fleets for this step [N, 4 players]
            arr = inc_matrix[:n_p, :4, b]
            
            # 3. Sort to find largest and second-largest arriving fleets
            arr_sorted = np.sort(arr, axis=1)
            top1_ships = arr_sorted[:, 3]
            top2_ships = arr_sorted[:, 2]

            survivor_ships = top1_ships - top2_ships
            top1_slot = np.argmax(arr, axis=1)
            
            # If there's a surviving arriving fleet, mark its owner. Otherwise 4 (nobody).
            survivor_slot = np.where(survivor_ships > 0, top1_slot, 4)

            # 4. Resolve combat between arriving survivor and garrison
            match = (survivor_slot == own_slot) & (survivor_slot != 4)
            conflict = (survivor_slot != own_slot) & (survivor_slot != 4)
            
            takeover = conflict & (survivor_ships > defenders)
            survive = conflict & (survivor_ships <= defenders)

            # Resolve identical ownership (reinforcements)
            defenders = np.where(match, defenders + survivor_ships, defenders)
            
            # Resolve takeovers
            own_slot = np.where(takeover, survivor_slot, own_slot)
            defenders = np.where(takeover, survivor_ships - defenders, defenders)
            
            # Resolve holds
            defenders = np.where(survive, defenders - survivor_ships, defenders)

            timeline_ships[:, b + 1] = defenders
            timeline_slots[:, b + 1] = own_slot

        return timeline_ships, timeline_slots, prod

    def _encode_edges(
        self,
        planets,
        step,
        comet_info,
        timeline_ships,
        timeline_slots,
        prod_rates,
        shared_cache: Optional[Dict[str, Any]] = None,
    ):
        n_p = min(len(planets), MAX_PLANETS)
        omega = float(self.angular_velocity or 0.0)
        
        self.edge_mask[:n_p, :n_p] = 1.0
        buf = self.edge_buf[:n_p, :n_p]

        p_ships = np.array([p[5] for p in planets[:n_p]], dtype=np.float32)
        p_owners_slots = np.array([self.owner_slot.get(int(p[1]), 4) for p in planets[:n_p]], dtype=np.int32)
        p_x = np.array([p[2] for p in planets[:n_p]], dtype=np.float32)
        p_y = np.array([p[3] for p in planets[:n_p]], dtype=np.float32)
        p_r = np.array([p[4] for p in planets[:n_p]], dtype=np.float32)
        p_pid = np.array([int(p[0]) for p in planets[:n_p]], dtype=np.int32)

        neutral_mask = (p_owners_slots == 4)

        # Blocked/invalid source rows get the final ETA bucket.
        buf[neutral_mask, :, _E_OFF_MAX_TTA_BUCKETS + EDGE_ETA_BUCKETS - 1] = 1.0
        buf[neutral_mask, :, _E_OFF_MAX_BLOCKED] = 1.0
        buf[neutral_mask, :, _E_OFF_CAN_TAKEOVER] = 0.0

        # Diagonal/self edges are canonical wait targets. They are not real
        # launches, but keep them finite and unblocked for stable masks.
        diag = np.arange(n_p)
        buf[diag, diag, _E_OFF_MAX_TTA_BUCKETS] = 1.0
        buf[diag, diag, _E_OFF_MAX_BLOCKED] = 0.0
        buf[diag, diag, _E_OFF_CAN_TAKEOVER] = 0.0

        i_idx, j_idx = np.where((p_owners_slots[:, None] != 4) & (np.arange(n_p)[:, None] != np.arange(n_p)[None, :]))
        if len(i_idx) == 0:
            return

        edge_key = None
        edge_geom = None

        if shared_cache is not None:
            # Exact, step-local board-geometry cache. This is intentionally
            # overwritten every new board step by the worker/offline worker,
            # so it cannot grow unbounded.
            planet_sig = tuple(
                (
                    int(p[0]),
                    int(p[1]),
                    round(float(p[2]), 6),
                    round(float(p[3]), 6),
                    round(float(p[4]), 6),
                    int(p[5]),
                    int(p[6]),
                )
                for p in planets[:n_p]
            )
            comet_sig = tuple(
                sorted(
                    (int(pid), int(info[1]), int(info[2]))
                    for pid, info in comet_info.items()
                )
            )
            edge_key = (int(step), planet_sig, comet_sig)

            if shared_cache.get("edge_geometry_key") == edge_key:
                edge_geom = shared_cache.get("edge_geometry")

        if edge_geom is not None:
            i_idx = edge_geom["i_idx"]
            j_idx = edge_geom["j_idx"]
            max_etas = edge_geom["max_etas"]
            max_blocked = edge_geom["max_blocked"]

        else:
            is_comet = np.isin(p_pid, list(comet_info.keys()))
            pair_is_comet_tgt = is_comet[j_idx]

            max_etas = np.full(len(i_idx), np.nan, dtype=np.float32)
            max_blocked = np.ones(len(i_idx), dtype=bool)

            needs_solve = ~pair_is_comet_tgt

            solve_idx = np.where(needs_solve)[0]
            if len(solve_idx) > 0:
                batch_i, batch_j = i_idx[solve_idx], j_idx[solve_idx]
                
                dx = p_x[batch_i] - p_x[batch_j]
                dy = p_y[batch_i] - p_y[batch_j]
                min_dist = np.maximum(0.0, np.hypot(dx, dy) - p_r[batch_j])
                reach_mask = (min_dist / self.fast_speed_const) <= EDGE_TTA_MAX
                
                valid_i = batch_i[reach_mask]
                valid_j = batch_j[reach_mask]
                valid_solve_idx = solve_idx[reach_mask]

                if len(valid_i) > 0:
                    src_xy = list(zip(p_x[valid_i], p_y[valid_i]))
                    tgt_xy = list(zip(p_x[valid_j], p_y[valid_j]))
                    radii = p_r[valid_j].tolist()

                    r_sun = np.hypot(p_x[valid_j] - SUN_X, p_y[valid_j] - SUN_Y)
                    orbit_flags = (
                        is_orbiting(r_sun, p_r[valid_j]) & (omega != 0.0)
                    ).tolist()

                    m_ships = np.maximum(
                        1,
                        p_ships[valid_i],
                    ).astype(np.int32)
    
                    m_arr = fleet_speed_np(m_ships.astype(np.int64))

                    edge_solve_horizon = min(
                        int(self.max_steps) - int(step),
                        int(math.ceil(EDGE_TTA_MAX)) + 2,
                    )
                    edge_solve_horizon = max(1, edge_solve_horizon)

                    ok_m, angle_m, t_m = batched_planet_intercepts(
                        src_xy,
                        tgt_xy,
                        m_arr,
                        radii,
                        orbit_flags,
                        omega=omega,
                        step=step,
                        max_turns=edge_solve_horizon,
                        device="cpu",
                    )

                    ok_m = ok_m.cpu().numpy().astype(bool)
                    angle_m = angle_m.cpu().numpy().astype(np.float32)
                    t_m = t_m.cpu().numpy().astype(np.float32)

                    sx = p_x[valid_i]
                    sy = p_y[valid_i]
                    sr = p_r[valid_i] + 0.1

                    def sun_blocks_candidate_batch(angles, spds, etas):
                        angles = np.asarray(angles, dtype=np.float32)
                        spds = np.asarray(spds, dtype=np.float32)
                        etas = np.asarray(etas, dtype=np.float32)

                        valid_eta = np.isfinite(etas) & (etas >= 0.0)
                        t = np.where(valid_eta, etas, 0.0).astype(np.float32)

                        lx = sx + np.cos(angles) * sr
                        ly = sy + np.sin(angles) * sr

                        ex = lx + np.cos(angles) * spds * t
                        ey = ly + np.sin(angles) * spds * t

                        vx = ex - lx
                        vy = ey - ly

                        wx = SUN_X - lx
                        wy = SUN_Y - ly

                        denom = vx * vx + vy * vy
                        denom = np.maximum(denom, 1e-9)

                        u = (wx * vx + wy * vy) / denom
                        u = np.clip(u, 0.0, 1.0)

                        cx = lx + u * vx
                        cy = ly + u * vy

                        d2 = (cx - SUN_X) * (cx - SUN_X) + (cy - SUN_Y) * (cy - SUN_Y)
                        return valid_eta & (d2 <= SUN_RADIUS * SUN_RADIUS)

                    m_sun_blocked = sun_blocks_candidate_batch(angle_m, m_arr, t_m)
                    m_good = ok_m & np.isfinite(t_m) & (t_m >= 0.0) & ~m_sun_blocked
    
                    m_eta_v = np.where(m_good, t_m, np.nan).astype(np.float32)
                    m_blk_v = ~m_good
       
                    max_etas[valid_solve_idx] = m_eta_v
                    max_blocked[valid_solve_idx] = m_blk_v

            # Comet-target edges use max-send timing only.
            #
            # We do the analytic intercept solve in a Python loop (cheap; just a
            # closure + 6-iter fixed point per pair), then *batch* the
            # replay-validated trace across every solved candidate in a single
            # ``_trace_fleets_numpy_engine_semantics`` call. The previous
            # implementation called the tracer once per (source, comet) pair,
            # paying the per-call broadcast/allocation overhead each time.
            comet_pair_idx = np.where(pair_is_comet_tgt)[0]
            if comet_pair_idx.size > 0:
                comet_remaining = self.max_steps - int(step)
                comet_horizon = max(1, min(comet_remaining, int(math.ceil(EDGE_TTA_MAX)) + 2))

                cand_idx: List[int] = []
                cand_fleets: List[List] = []
                cand_speeds: List[float] = []
                cand_tgt_pids: List[int] = []

                for idx in comet_pair_idx:
                    i = int(i_idx[idx])
                    j = int(j_idx[idx])
                    src_pl = planets[i]
                    tgt_pl = planets[j]

                    m_s = max(1, int(src_pl[5]))
                    speed = fleet_speed(m_s)
                    if speed <= 0.0:
                        max_etas[idx] = np.nan
                        max_blocked[idx] = True
                        continue

                    src_center = (float(src_pl[2]), float(src_pl[3]))
                    src_r = float(src_pl[4])

                    make_solver = self._make_target_solver(tgt_pl, step, comet_info)
                    solved = _solve_with_launch_offset(
                        src_center, src_r, make_solver(speed),
                    )
                    if solved is None or not math.isfinite(float(solved[0])):
                        max_etas[idx] = np.nan
                        max_blocked[idx] = True
                        continue

                    angle = float(solved[0])
                    lx = src_center[0] + math.cos(angle) * (src_r + 0.1)
                    ly = src_center[1] + math.sin(angle) * (src_r + 0.1)

                    cand_idx.append(int(idx))
                    cand_fleets.append([-1, -1, lx, ly, angle, -1, m_s])
                    cand_speeds.append(float(speed))
                    cand_tgt_pids.append(int(tgt_pl[0]))

                if cand_fleets:
                    speeds_arr = np.array(cand_speeds, dtype=np.float32)
                    hit_pids_c, etas_c, blocked_c = self._trace_fleets_numpy_engine_semantics(
                        cand_fleets,
                        planets=planets[:n_p],
                        current_step=int(step),
                        comet_info=comet_info,
                        max_turns=comet_horizon,
                        speeds=speeds_arr,
                    )

                    cand_idx_arr = np.asarray(cand_idx, dtype=np.int64)
                    cand_tgt_arr = np.asarray(cand_tgt_pids, dtype=np.int64)
                    good = (
                        (~blocked_c)
                        & np.isfinite(etas_c)
                        & (hit_pids_c.astype(np.int64) == cand_tgt_arr)
                    )

                    max_etas[cand_idx_arr] = np.where(good, etas_c, np.nan).astype(np.float32)
                    max_blocked[cand_idx_arr] = ~good

            if shared_cache is not None and edge_key is not None:
                shared_cache["edge_geometry_key"] = edge_key
                shared_cache["edge_geometry"] = {
                    "i_idx": i_idx.copy(),
                    "j_idx": j_idx.copy(),
                    "max_etas": max_etas.copy(),
                    "max_blocked": max_blocked.copy(),
                }

        def vec_project_state(tgts, etas, blockeds):
            """
            Project target garrison and owner slot at the max-send ETA.

            Returns:
              projected_ships: target garrison at arrival
              projected_slots: owner slot at arrival, perspective-relative
            """
            projected_ships = p_ships[tgts].copy()
            projected_slots = p_owners_slots[tgts].copy()

            valid = ~blockeds & ~np.isnan(etas)
            if np.any(valid):
                v_tgts = tgts[valid]
                v_etas = etas[valid]
                eta_int = np.ceil(v_etas).astype(np.int32)

                in_h = eta_int <= TTA_BUCKETS

                out_ships = np.zeros(len(v_tgts), dtype=np.float32)
                out_slots = np.zeros(len(v_tgts), dtype=np.int32)

                if np.any(in_h):
                    out_ships[in_h] = timeline_ships[v_tgts[in_h], eta_int[in_h]]
                    out_slots[in_h] = timeline_slots[v_tgts[in_h], eta_int[in_h]]

                if np.any(~in_h):
                    o_tgts = v_tgts[~in_h]
                    o_etas = eta_int[~in_h]

                    base_ships = timeline_ships[o_tgts, TTA_BUCKETS]
                    base_slots = timeline_slots[o_tgts, TTA_BUCKETS]

                    # Production after the explicit timeline horizon accrues
                    # only for owned planets. Neutral slot 4 does not produce.
                    extra = np.where(
                        base_slots != 4,
                        prod_rates[o_tgts] * (o_etas - TTA_BUCKETS),
                        0.0,
                    )

                    out_ships[~in_h] = base_ships + extra
                    out_slots[~in_h] = base_slots

                projected_ships[valid] = out_ships
                projected_slots[valid] = out_slots

            return projected_ships, projected_slots

        src_slot = p_owners_slots[i_idx]
        tgt_slot = p_owners_slots[j_idx]
        src_ships = p_ships[i_idx]

        proj_ships, proj_slots = vec_project_state(j_idx, max_etas, max_blocked)

        valid_attack = (
            (~max_blocked)
            & np.isfinite(max_etas)
            & (src_slot != 4)
            & (src_slot != proj_slots)
        )

        # Engine combat takeover requires attacking ships > defending ships.
        can_takeover = (valid_attack & (src_ships > proj_ships)).astype(np.float32)

        # Write max-send edge features.
        eta_oh = self._eta_onehot_matrix(max_etas, max_blocked)
        buf[
            i_idx,
            j_idx,
            _E_OFF_MAX_TTA_BUCKETS : _E_OFF_MAX_TTA_BUCKETS + EDGE_ETA_BUCKETS,
        ] = eta_oh

        buf[i_idx, j_idx, _E_OFF_MAX_BLOCKED] = max_blocked.astype(np.float32)
        buf[i_idx, j_idx, _E_OFF_CAN_TAKEOVER] = can_takeover

        # Write Edge Relationship Tags
        buf[i_idx, j_idx, _E_OFF_SRC_OWNER + src_slot] = 1.0

        buf[i_idx, j_idx, _E_OFF_TGT_OWNER + tgt_slot] = 1.0

    def assemble(
        self,
        obs,
        step,
        compute_expert=False,
        shared_cache: Optional[Dict[str, Any]] = None,
    ):
        if self.player_id is None:
            self.reset(obs, int(obs.get("player", 0)))

        self.planet_buf.fill(0.0)
        self.planet_mask.fill(0.0)
        self.global_buf.fill(0.0)
        self.source_mask.fill(0.0)
        self.edge_buf.fill(0.0)
        self.edge_mask.fill(0.0)

        planets = obs.get("planets", []) or []
        fleets = obs.get("fleets", []) or []
        comets = obs.get("comets", []) or []

        comet_info = {}
        current_comets = set()
        for grp in comets:
            pids = grp.get("planet_ids", [])
            paths = grp.get("paths", [])
            pidx = int(grp.get("path_index", 0))
            for body_idx, pid in enumerate(pids):
                current_comets.add(int(pid))
                if body_idx < len(paths):
                    comet_info[int(pid)] = (paths[body_idx], pidx, len(paths[body_idx]))

        if shared_cache is None:
            fleet_radar = self.fleet_radar
            known_comets = self.known_comets
        else:
            fleet_radar = shared_cache.setdefault("fleet_radar", {})
            known_comets = shared_cache.get("known_comets", set())

        if current_comets != known_comets:
            fleet_radar.clear()
            if shared_cache is None:
                self.known_comets = current_comets
            else:
                shared_cache["known_comets"] = set(current_comets)

        current_fids = {int(f[0]) for f in fleets}
        dead_fids = [fid for fid in fleet_radar if int(fid) not in current_fids]
        for fid in dead_fids:
            del fleet_radar[fid]

        # --- PROCESS FLEETS INTO TTA BUCKETS (4-PLAYER MATRIX) ---
        inc_matrix = np.zeros((MAX_PLANETS, 4, TTA_BUCKETS), dtype=np.float32)
        pid_to_idx = {int(p[0]): i for i, p in enumerate(planets[:MAX_PLANETS])}

        # Project all newly observed fleets together using the same validated
        # NumPy collision-selection kernel as the simulator. Cached entries are
        # retained until the fleet disappears, or until the visible comet set
        # changes and the future obstacle field must be recomputed.
        missing_fleets = [f for f in fleets if int(f[0]) not in fleet_radar]
        if missing_fleets:
            # Incoming arrivals are clamped into ``TTA_BUCKETS`` below, so
            # projecting beyond ``TTA_BUCKETS + 2`` turns is wasted compute -
            # the policy sees the same observation either way. The +2 keeps a
            # small safety margin for fractional arrival times.
            remaining = self.max_steps - int(step)
            inc_horizon = max(0, min(remaining, TTA_BUCKETS + 2))
            target_pids, etas, blocked = self._trace_fleets_numpy_engine_semantics(
                missing_fleets,
                planets=planets[:MAX_PLANETS],
                current_step=int(step),
                comet_info=comet_info,
                max_turns=inc_horizon,
            )
        
            for f, target_pid, eta, is_blocked in zip(missing_fleets, target_pids, etas, blocked):
                fid = int(f[0])
                if bool(is_blocked) or int(target_pid) < 0 or not np.isfinite(eta):
                    fleet_radar[fid] = (None, -1)
                    continue
        
                eta_f = max(0.0, float(eta))
                arrival_delta = max(1, int(math.ceil(eta_f - 1e-9)))
                arrival_step = int(step) + arrival_delta
        
                # Strategic-planning semantics:
                # If currently visible geometry predicts a planet collision,
                # encode that likely incoming. A future comet spawn does not
                # invalidate it until the comet is actually visible and confirms
                # an intercept. At that spawn frame, the comet set changes,
                # fleet_radar clears, and the fleet is reprojected.
                fleet_radar[fid] = (int(target_pid), int(arrival_step))

        for f in fleets:
            fid, f_owner, _x, _y, _a, _from, ships = f
            target_pid, arrival_step = fleet_radar.get(int(fid), (None, -1))
            
            if target_pid is not None and target_pid in pid_to_idx:
                idx = pid_to_idx[target_pid]
                # inc_matrix[..., 0] is applied during the transition step -> step+1.
                # The cache stores the produced-frame step number, so subtract 1.
                tta = int(arrival_step) - int(step)
                bucket = max(0, min(tta - 1, TTA_BUCKETS - 1))
                
                owner_slot = self.owner_slot.get(int(f_owner), 4)
                if owner_slot < 4:
                    inc_matrix[idx, owner_slot, bucket] += int(ships)

        n_p = min(len(planets), MAX_PLANETS)

        # 1. Simulate exact combat for the entire TTA horizon
        timeline_ships, timeline_slots, prod_rates = self._build_garrison_timeline(planets[:n_p], inc_matrix)

        # 2. Encode planets (vectorized: writes into ``self.planet_buf``,
        #    ``self.planet_mask`` and ``self.source_mask`` in one pass).
        self._encode_planets_vec(
            planets[:n_p],
            step=step,
            comet_info=comet_info,
            inc_matrix=inc_matrix,
            timeline_ships=timeline_ships,
            timeline_slots=timeline_slots,
        )
        
        # 3. O(1) Timeline Projection fallback for edges
        self._encode_edges(
            planets[:MAX_PLANETS],
            step,
            comet_info,
            timeline_ships,
            timeline_slots,
            prod_rates,
            shared_cache=shared_cache,
        )        
        # 4. Global encodings
        self._encode_global(obs, step, len(planets), len(fleets))

        expert_action = None
        if compute_expert:
            ally_inc = inc_matrix[:, 0, :]
            enemy_inc = np.sum(inc_matrix[:, 1:4, :], axis=1)
            expert_action = self._compute_expert_action(planets[:MAX_PLANETS], step, ally_inc, enemy_inc, comet_info)

        # NOTE: the returned arrays are *views* into the assembler's persistent
        # buffers. They are valid only until the next ``assemble()`` call on the
        # same instance. All existing callers (``worker.pack_observation_np``,
        # ``offline_worker`` via ``torch.tensor``, ``audit_obs_replay_encoding``
        # via ``check_assembled_finite``) consume the dict synchronously before
        # the next assemble, so the previous defensive ``.copy()`` per buffer
        # was pure overhead - each step paid 6 full-buffer memcopies that the
        # downstream packer then copied again.
        return {
            "planet_features": self.planet_buf,
            "planet_mask": self.planet_mask,
            "edge_features": self.edge_buf,
            "edge_mask": self.edge_mask,
            "global_features": self.global_buf,
            "source_mask": self.source_mask,
            "n_planets_raw": len(planets),
            "n_fleets_raw": len(fleets),
            "n_comets_raw": len(comets),
            "expert_action": expert_action,
        }

    def _encode_planets_vec(
        self,
        planets,
        *,
        step: int,
        comet_info,
        inc_matrix: np.ndarray,
        timeline_ships: np.ndarray,
        timeline_slots: np.ndarray,
    ):
        """Vectorized planet token encoder.

        Writes ``self.planet_buf[:n_p]``, ``self.planet_mask[:n_p]`` and
        ``self.source_mask[:n_p]`` in one pass. Equivalent in shape and
        semantics to ``_encode_planet`` invoked in a Python ``for`` loop, but
        the per-bucket / per-flag / per-planet writes happen in NumPy.

        Buffers are assumed pre-zeroed by ``assemble``.
        """
        n_p = min(len(planets), MAX_PLANETS)
        if n_p <= 0:
            return

        buf = self.planet_buf  # [MAX_PLANETS, PLANET_DIM]
        rows = np.arange(n_p)

        # ------------------------------------------------------------
        # Single Python pass over planets to extract scalars + body kind.
        # ------------------------------------------------------------
        pids = np.empty(n_p, dtype=np.int64)
        owners = np.empty(n_p, dtype=np.int64)
        xs = np.empty(n_p, dtype=np.float32)
        ys = np.empty(n_p, dtype=np.float32)
        ships = np.empty(n_p, dtype=np.float32)
        prods = np.empty(n_p, dtype=np.int64)

        is_comet_mask = np.zeros(n_p, dtype=bool)
        is_orb_mask = np.zeros(n_p, dtype=bool)
        is_home_mask = np.zeros(n_p, dtype=bool)
        steps_left = np.zeros(n_p, dtype=np.int64)
        plen_arr = np.zeros(n_p, dtype=np.int64)

        dx = np.zeros(n_p, dtype=np.float32)
        dy = np.zeros(n_p, dtype=np.float32)
        owner_slots = np.empty(n_p, dtype=np.int64)

        angular_velocity = float(self.angular_velocity or 0.0)
        player_id = int(self.player_id) if self.player_id is not None else -1

        for i, p in enumerate(planets[:n_p]):
            pid = int(p[0])
            owner = int(p[1])
            pids[i] = pid
            owners[i] = owner
            xs[i] = float(p[2])
            ys[i] = float(p[3])
            ships_i = int(p[5])
            ships[i] = float(max(ships_i, 0))
            prods[i] = int(p[6])
            owner_slots[i] = self.owner_slot.get(owner, 4)
            is_home_mask[i] = pid in self.home_planet_ids

            if pid in comet_info:
                is_comet_mask[i] = True
                path, k, plen = comet_info[pid]
                plen_arr[i] = int(plen)
                steps_left[i] = max(0, plen - 1 - k)
                if k + 1 < plen:
                    nx, ny = path[k + 1]
                    dx[i] = float(nx) - xs[i]
                    dy[i] = float(ny) - ys[i]
            else:
                orb = bool(self.is_orbiting_cache.get(pid, False))
                is_orb_mask[i] = orb
                if orb and step >= 1:
                    r = float(self.r0[pid])
                    theta0 = float(self.theta_0[pid])
                    nx, ny = orbit_position(theta0, r, step + 1, angular_velocity)
                    dx[i] = float(nx) - xs[i]
                    dy[i] = float(ny) - ys[i]

            # Planet mask is always 1 inside [:n_p]; source mask requires
            # ownership and at least one ship.
            self.planet_mask[i] = 1.0
            if owner == player_id and ships_i >= 1:
                self.source_mask[i] = 1.0

        # ------------------------------------------------------------
        # Vectorized writes into planet_buf.
        # ------------------------------------------------------------
        buf[:n_p, _P_OFF_X] = xs / 100.0
        buf[:n_p, _P_OFF_Y] = ys / 100.0
        buf[:n_p, _P_OFF_DX] = dx / 10.0
        buf[:n_p, _P_OFF_DY] = dy / 10.0

        # ln(1 + ships) raw garrison.
        buf[:n_p, _P_OFF_SHIPS] = np.log1p(ships)

        # Production one-hot (clamped to [0, 4]).
        prod_idx = np.clip(prods - 1, 0, 4).astype(np.int64)
        buf[rows, _P_OFF_PROD + prod_idx] = 1.0

        # Owner one-hot.
        buf[rows, _P_OFF_OWNER + owner_slots] = 1.0

        # Flags. ``is_static`` is "neither comet nor orbiting" so the static
        # bit fires for planets sitting outside the orbital ring or when
        # omega == 0 (orbital cache stays True but the body doesn't move).
        is_static = ~is_comet_mask & ~is_orb_mask
        is_orb_only = ~is_comet_mask & is_orb_mask
        despawn_soon = is_comet_mask & (steps_left <= 5)

        buf[:n_p, _P_OFF_FLAGS + 0] = is_orb_only.astype(np.float32)
        buf[:n_p, _P_OFF_FLAGS + 1] = is_static.astype(np.float32)
        buf[:n_p, _P_OFF_FLAGS + 2] = is_comet_mask.astype(np.float32)
        buf[:n_p, _P_OFF_FLAGS + 3] = is_home_mask.astype(np.float32)
        buf[:n_p, _P_OFF_FLAGS + 4] = despawn_soon.astype(np.float32)

        # Despawn norm only matters for comets.
        if bool(is_comet_mask.any()):
            denom = np.maximum(1, plen_arr - 1).astype(np.float32)
            despawn = np.zeros(n_p, dtype=np.float32)
            np.divide(
                steps_left.astype(np.float32),
                denom,
                out=despawn,
                where=is_comet_mask,
            )
            buf[:n_p, _P_OFF_DESPAWN] = despawn

        # Raw 4-player TTA buckets: log1p(inc_matrix) flattened per planet.
        buf[:n_p, _P_OFF_INC : _P_OFF_INC + 4 * TTA_BUCKETS] = np.log1p(
            inc_matrix[:n_p].reshape(n_p, -1)
        )

        # Multi-way combat projection: garrison and projected owner.
        # ``timeline_ships`` and ``timeline_slots`` are sized [n_p, TTA_BUCKETS+1];
        # column 0 is the current frame, columns 1.. are the post-tick projection.
        buf[:n_p, _P_OFF_GARRISON : _P_OFF_GARRISON + TTA_BUCKETS] = np.log1p(
            np.maximum(timeline_ships[:n_p, 1:], 0.0)
        )

        # Projected-owner one-hot via advanced indexing:
        #     buf[i, _P_OFF_PROJ_OWNER + b*5 + slot] = 1.0
        bucket_rows = rows[:, None]
        bucket_cols = np.arange(TTA_BUCKETS, dtype=np.int64)[None, :]
        proj_slots = timeline_slots[:n_p, 1:].astype(np.int64)
        proj_col_offsets = _P_OFF_PROJ_OWNER + bucket_cols * 5 + proj_slots
        buf[bucket_rows, proj_col_offsets] = 1.0

    def _encode_planet(self, buf, planet, step, comet_info, inc_matrix_i, t_ships, t_slots):
        pid, owner, x, y, _radius, ships, prod = planet
        is_comet = pid in comet_info

        if is_comet:
            path, k, plen = comet_info[pid]
            if k + 1 < plen:
                nx, ny = path[k + 1]
                dx = nx - x
                dy = ny - y
            else:
                dx = dy = 0.0
            steps_left = max(0, plen - 1 - k)
        elif self.is_orbiting_cache.get(pid, False) and step >= 1:
            r = self.r0[pid]
            theta0 = self.theta_0[pid]
            nx, ny = orbit_position(theta0, r, step + 1, self.angular_velocity)
            dx = nx - x
            dy = ny - y
            steps_left = 0
        else:
            dx = dy = 0.0
            steps_left = 0

        buf[_P_OFF_X] = x / 100.0
        buf[_P_OFF_Y] = y / 100.0
        buf[_P_OFF_DX] = dx / 10.0
        buf[_P_OFF_DY] = dy / 10.0

        # --- RAW GARRISON ---
        buf[_P_OFF_SHIPS] = math.log1p(max(int(ships), 0))

        prod_idx = max(0, min(int(prod) - 1, 4))
        buf[_P_OFF_PROD + prod_idx] = 1.0

        slot = self.owner_slot.get(int(owner), 4)
        buf[_P_OFF_OWNER + slot] = 1.0

        is_orb = (not is_comet) and self.is_orbiting_cache.get(pid, False)
        is_static = (not is_comet) and not is_orb
        is_home = pid in self.home_planet_ids
        is_despawning_soon = is_comet and steps_left <= 5
        
        if is_orb: buf[_P_OFF_FLAGS + 0] = 1.0
        if is_static: buf[_P_OFF_FLAGS + 1] = 1.0
        if is_comet: buf[_P_OFF_FLAGS + 2] = 1.0
        if is_home: buf[_P_OFF_FLAGS + 3] = 1.0
        if is_despawning_soon: buf[_P_OFF_FLAGS + 4] = 1.0

        if is_comet:
            _, _, plen = comet_info[pid]
            buf[_P_OFF_DESPAWN] = steps_left / max(1, plen - 1)
            
        # --- RAW 4-PLAYER TTA BUCKETS ---
        buf[_P_OFF_INC : _P_OFF_INC + (4 * TTA_BUCKETS)] = np.log1p(inc_matrix_i.flatten())

        # --- MULTI-WAY COMBAT PROJECTIONS ---
        for b in range(TTA_BUCKETS):
            s_ships = t_ships[b + 1]
            s_slot = t_slots[b + 1]
            
            # Projected garrison and owner after incoming-fleet combat.
            buf[_P_OFF_GARRISON + b] = math.log1p(max(float(s_ships), 0.0))
            buf[_P_OFF_PROJ_OWNER + (b * 5) + int(s_slot)] = 1.0

    def _encode_global(self, obs, step, n_planets, n_fleets):
        self.global_buf[0] = float(step) / float(self.max_steps)

        planet_ships = np.zeros(5)
        planet_prod = np.zeros(5)
        my_planets = opp_planets = neutral_planets = 0

        for p in obs.get("planets", []) or []:
            owner = int(p[1])
            slot = self.owner_slot.get(owner, 4)
            planet_ships[slot] += int(p[5])
            planet_prod[slot] += int(p[6])
            
            if owner == self.player_id:
                my_planets += 1
            elif owner == -1:
                neutral_planets += 1
            else:
                opp_planets += 1

        fleet_ships = np.zeros(5)
        for f in obs.get("fleets", []) or []:
            slot = self.owner_slot.get(int(f[1]), 4)
            fleet_ships[slot] += int(f[6])

        tot_ships = planet_ships + fleet_ships

        my_total = math.log1p(tot_ships[0])
        opp_total = math.log1p(np.sum(tot_ships[1:4]))
        ship_diff = my_total - opp_total  # positive = winning

        next_event = next((c for c in COMET_SPAWN_STEPS if c >= step), self.max_steps)
        steps_to_next_comet = max(0, next_event - step)
        next_comet_norm = min(1.0, steps_to_next_comet / 100.0)

        off = 1
        
        # Original 13 Features
        self.global_buf[off + 0] = (self.angular_velocity or 0.0) / 0.05
        self.global_buf[off + 1] = next_comet_norm
        self.global_buf[off + 2] = my_total / 12.0
        self.global_buf[off + 3] = opp_total / 12.0
        self.global_buf[off + 4] = ship_diff / 12.0
        self.global_buf[off + 5] = my_planets / float(MAX_PLANETS)
        self.global_buf[off + 6] = opp_planets / float(MAX_PLANETS)
        self.global_buf[off + 7] = neutral_planets / float(MAX_PLANETS)
        self.global_buf[off + 8] = n_fleets / float(MAX_FLEETS)
        for k in range(MAX_PLAYERS):
            self.global_buf[off + 9 + k] = 1.0 if self.player_id == k else 0.0

        # New 17 Log Features (to be two-hot encoded by schema_metadata)
        for p in range(4):
            self.global_buf[off + 13 + p] = math.log1p(tot_ships[p])
            self.global_buf[off + 17 + p] = math.log1p(planet_prod[p])
            self.global_buf[off + 21 + p] = math.log1p(fleet_ships[p])
            self.global_buf[off + 25 + p] = math.log1p(planet_ships[p])
            
        self.global_buf[off + 29] = float(steps_to_next_comet)

    def map_actions_to_orders(
        self,
        target_idx,
        option_idx,
        obs,
        step: Optional[int] = None,
        allow_exact_fallback: bool = True,
    ):
        """Convert (target, option) per-source actions into engine orders.

        The hot path is split into three passes:

          1. Per-source analytic angle solve (Python, but no trace yet).
          2. One batched ``_trace_fleets_numpy_engine_semantics`` call over
             all analytic-solved candidates - replaces the per-source scalar
             traces that previously dominated this method.
          3. Optional exact-angle fallback for any unresolved sources, used
             only by offline replay audits (``allow_exact_fallback=True``).
             PPO calls with ``allow_exact_fallback=False`` so the entire
             fallback pass is skipped.
        """
        orders: List[List] = []
        planets = (obs.get("planets") or [])[:MAX_PLANETS]

        comet_paths: Dict[int, Tuple[List, int]] = {}
        comet_info: Dict[int, Tuple[List, int, int]] = {}
        for grp in (obs.get("comets") or []):
            pids = grp.get("planet_ids") or []
            paths = grp.get("paths") or []
            pidx = int(grp.get("path_index") or 0)
            for body_idx, pid in enumerate(pids):
                if body_idx < len(paths):
                    path = paths[body_idx]
                    comet_paths[int(pid)] = (path, pidx)
                    comet_info[int(pid)] = (path, pidx, len(path))

        if step is None:
            cur_step = int(obs.get("step_number", obs.get("step", 1)) or 0)
        else:
            cur_step = int(step)

        ti = np.asarray(target_idx, dtype=np.int64)
        oi = np.asarray(option_idx, dtype=np.int64)

        # Edge ETAs and action solves only matter inside ``EDGE_TTA_MAX``;
        # any candidate not validated within that horizon is dropped.
        remaining = self.max_steps - cur_step
        action_horizon = max(1, min(remaining, int(math.ceil(EDGE_TTA_MAX)) + 2))

        # ------------------------------------------------------------
        # Pass 1: per-source analytic angle solve.
        # ------------------------------------------------------------
        candidates: List[Dict[str, Any]] = []
        fallback_pending: List[Dict[str, Any]] = []

        for i, src in enumerate(planets):
            src_pid, src_owner, src_x, src_y, src_radius, src_ships, _src_prod = src
            if int(src_owner) != self.player_id:
                continue
            src_ships = int(src_ships)
            if src_ships < 1:
                continue
            if int(oi[i]) != OPT_MAX_SEND:
                continue

            t = int(ti[i])
            if t < 0 or t >= len(planets) or t == i:
                continue

            # Fast reject impossible policy-selected edges before running
            # analytic intercept solving or the expensive exact angle fallback.
            try:
                if float(self.edge_buf[i, t, _E_OFF_MAX_BLOCKED]) > 0.5:
                    continue
            except Exception:
                pass

            tgt = planets[t]
            tgt_pid = int(tgt[0])
            tgt_x = float(tgt[2])
            tgt_y = float(tgt[3])
            tgt_radius = float(tgt[4])

            src_center = (float(src_x), float(src_y))
            src_radius_f = float(src_radius)
            ship_count = src_ships
            speed = fleet_speed(ship_count)
            if speed <= 0.0:
                continue

            # Bind loop variables via default args so closures stay correct
            # even if we ever defer the call across iterations.
            if tgt_pid in comet_paths:
                _path, _pidx = comet_paths[tgt_pid]
                def make_solver(spd, _path=_path, _pidx=_pidx, _tr=tgt_radius):
                    def solve_from_origin(origin):
                        return lead_intercept_path(
                            origin, _path, _pidx, spd, hit_radius=_tr
                        )
                    return solve_from_origin
            elif tgt_pid in self.planet_paths:
                _path = self.planet_paths[tgt_pid]
                def make_solver(spd, _path=_path, _cs=cur_step, _tr=tgt_radius):
                    def solve_from_origin(origin):
                        return lead_intercept_path(
                            origin, _path, _cs, spd, hit_radius=_tr
                        )
                    return solve_from_origin
            else:
                def make_solver(spd, _tx=tgt_x, _ty=tgt_y):
                    def solve_from_origin(origin):
                        return lead_intercept(origin, (_tx, _ty), (0.0, 0.0), spd)
                    return solve_from_origin

            solved = _solve_with_launch_offset(
                src_center, src_radius_f, make_solver(speed),
            )

            if solved is None or not math.isfinite(float(solved[0])):
                if allow_exact_fallback:
                    fallback_pending.append({
                        "src_pid": int(src_pid),
                        "src_center": src_center,
                        "src_radius": src_radius_f,
                        "ship_count": ship_count,
                        "tgt_pid": tgt_pid,
                        "seed_angle": None,
                    })
                continue

            angle = float(solved[0])
            launch_x = src_center[0] + math.cos(angle) * (src_radius_f + 0.1)
            launch_y = src_center[1] + math.sin(angle) * (src_radius_f + 0.1)

            candidates.append({
                "src_pid": int(src_pid),
                "src_center": src_center,
                "src_radius": src_radius_f,
                "angle": angle,
                "launch_x": launch_x,
                "launch_y": launch_y,
                "speed": float(speed),
                "ship_count": ship_count,
                "tgt_pid": tgt_pid,
            })

        # ------------------------------------------------------------
        # Pass 2: one batched verification trace over all candidates.
        # ------------------------------------------------------------
        if candidates:
            fleets_batch = [
                [-1, -1, c["launch_x"], c["launch_y"], c["angle"], -1, c["ship_count"]]
                for c in candidates
            ]
            speeds_batch = np.array(
                [c["speed"] for c in candidates], dtype=np.float32,
            )

            hit_pids_b, etas_b, blocked_b = self._trace_fleets_numpy_engine_semantics(
                fleets_batch,
                planets=planets,
                current_step=cur_step,
                comet_info=comet_info,
                max_turns=action_horizon,
                speeds=speeds_batch,
            )

            for k, c in enumerate(candidates):
                ok = (
                    (not bool(blocked_b[k]))
                    and int(hit_pids_b[k]) == int(c["tgt_pid"])
                    and bool(np.isfinite(etas_b[k]))
                )
                if ok:
                    orders.append(
                        [c["src_pid"], float(c["angle"]), int(c["ship_count"])]
                    )
                elif allow_exact_fallback:
                    fallback_pending.append({
                        "src_pid": c["src_pid"],
                        "src_center": c["src_center"],
                        "src_radius": c["src_radius"],
                        "ship_count": c["ship_count"],
                        "tgt_pid": c["tgt_pid"],
                        "seed_angle": c["angle"],
                    })

        # ------------------------------------------------------------
        # Pass 3: exact 1440-angle fallback (offline audit only).
        # ------------------------------------------------------------
        if allow_exact_fallback and fallback_pending:
            for fb in fallback_pending:
                fallback = self._scan_angle_to_target_exact(
                    src_center=fb["src_center"],
                    src_radius=fb["src_radius"],
                    ship_count=int(fb["ship_count"]),
                    tgt_pid=int(fb["tgt_pid"]),
                    planets=planets,
                    current_step=cur_step,
                    comet_info=comet_info,
                    max_turns=self.max_steps - cur_step,
                    seed_angle=fb["seed_angle"],
                )
                if fallback is None:
                    continue
                angle_fb, _ = fallback
                orders.append(
                    [int(fb["src_pid"]), float(angle_fb), int(fb["ship_count"])]
                )

        return orders


def schema_metadata():
    """Return dim/cap info for downstream model wiring."""
    
    # Base offsets for dynamic global features.
    # global[0] is normalized step; global[1:31] are the 30 scalar features.
    off = 1
    _G_OFF_LN_TOT_SHIPS = off + 13
    _G_OFF_LN_PROD = off + 17
    _G_OFF_LN_FLEETS = off + 21
    _G_OFF_LN_PLANETS = off + 25
    _G_OFF_NEXT_COMET = off + 29
    
    return {
        "n_bins": N_BINS,
        "max_planets": MAX_PLANETS,
        "max_players": MAX_PLAYERS,
        "planet_dim": PLANET_DIM,
        "global_dim": GLOBAL_DIM,
        "planet_offsets": {
            "x": _P_OFF_X,
            "y": _P_OFF_Y,
            "dx": _P_OFF_DX,
            "dy": _P_OFF_DY,
            "ships": _P_OFF_SHIPS,
            "prod": _P_OFF_PROD,
            "owner": _P_OFF_OWNER,
            "flags": _P_OFF_FLAGS,
            "despawn": _P_OFF_DESPAWN,
            "inc": _P_OFF_INC,
            "garrison": _P_OFF_GARRISON,
            "projected_owner": _P_OFF_PROJ_OWNER,
        },
        "planet_twohot_specs": [
            {"start": _P_OFF_SHIPS, "length": 1, "min": LOG_SHIPS_LOW, "max": LOG_SHIPS_HIGH, "bins": N_SHIP_BINS},
            {"start": _P_OFF_INC, "length": 4 * TTA_BUCKETS, "min": LOG_SHIPS_LOW, "max": LOG_SHIPS_HIGH, "bins": N_SHIP_BINS},
            {"start": _P_OFF_GARRISON, "length": TTA_BUCKETS, "min": LOG_SHIPS_LOW, "max": LOG_SHIPS_HIGH, "bins": N_SHIP_BINS},
        ],
        "global_twohot_specs": [
            {"start": _G_OFF_LN_TOT_SHIPS, "length": 4, "min": LOG_SHIPS_LOW, "max": LOG_SHIPS_HIGH, "bins": 51},
            {"start": _G_OFF_LN_PROD, "length": 4, "min": 0.0, "max": 6.0, "bins": 51},
            {"start": _G_OFF_LN_FLEETS, "length": 4, "min": LOG_SHIPS_LOW, "max": LOG_SHIPS_HIGH, "bins": 51},
            {"start": _G_OFF_LN_PLANETS, "length": 4, "min": LOG_SHIPS_LOW, "max": LOG_SHIPS_HIGH, "bins": 51},
            {"start": _G_OFF_NEXT_COMET, "length": 1, "min": 0.0, "max": 500.0, "bins": 51},
        ],
        "edge_dim": EDGE_DIM,
        "edge_offsets": {
            "max_eta": _E_OFF_MAX_TTA,
            "max_eta_buckets": _E_OFF_MAX_TTA_BUCKETS,
            "max_eta_bucket_count": EDGE_ETA_BUCKETS,
            "max_blocked": _E_OFF_MAX_BLOCKED,
            "can_takeover": _E_OFF_CAN_TAKEOVER,
        },
        "edge_twohot_specs": [],
        "edge_tta_max": EDGE_TTA_MAX,
        "edge_eta_buckets": EDGE_ETA_BUCKETS,
        "edge_blocked_tta_norm": EDGE_BLOCKED_TTA_NORM,
        "n_production": 5,
        "n_owner_slots_planet": 5,
        "n_owner_slots_fleet": 4,
        "n_planet_flags": 5,
        "tta_buckets": TTA_BUCKETS,
        "n_target_buckets": MAX_PLANETS,
        "n_action_options": N_ACTION_OPTIONS,
        "n_option_buckets": N_ACTION_OPTIONS,
        "action_dim_per_planet": MAX_PLANETS * N_ACTION_OPTIONS,
    }
