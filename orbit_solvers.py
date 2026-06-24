"""
Lead-targeting solvers for Orbit Wars (v2 -- supersedes solvers.py).

This file replaces solvers.py to dodge a workspace mount-sync bug
that capped the prior file's writable size mid-edit during the
hit_radius / step / freeze fix. Imports should target this module:

    from orbit_solvers import (
        lead_intercept,
        lead_intercept_orbit,
        lead_intercept_path,
    )

`solvers.py`, `intercept.py`, and `targeting.py` re-export from here.

Three solvers, one per body type:

  * `lead_intercept`       - closed-form for *constant-velocity*
                             targets (static planets, fleets).
  * `lead_intercept_orbit` - orbiting planets. Walks the actual rotation
                             around the sun. Now models the engine's
                             one-frame freeze when launching at step 0,
                             and uses a per-planet hit_radius so the
                             solver only returns angles the fleet can
                             physically land on.
  * `lead_intercept_path`  - comets. Iterates the engine's precomputed
                             trajectory.

All three return `(angle_radians, t_turns)` or None.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

# Mirror engine constants so this module is standalone.
_SUN_X, _SUN_Y = 50.0, 50.0
_EPS = 1e-9


def lead_intercept(
    src: Tuple[float, float],
    tgt_pos: Tuple[float, float],
    tgt_vel: Tuple[float, float],
    fleet_speed: float,
) -> Optional[Tuple[float, float]]:
    """
    Solve `|| (p_t - p_s) + v_t * t || = s * t` for smallest t > 0.

    Squaring gives `(s^2 - |v_t|^2) t^2 - 2(r.v_t) t - |r|^2 = 0`.
    Returns None when target flees faster than fleet, fleet_speed <= 0,
    or both quadratic roots are non-positive.
    """
    if fleet_speed <= 0.0:
        return None

    rx = tgt_pos[0] - src[0]
    ry = tgt_pos[1] - src[1]
    vx, vy = tgt_vel

    r2 = rx * rx + ry * ry
    if r2 < _EPS:
        return 0.0, 0.0

    v2 = vx * vx + vy * vy
    rv = rx * vx + ry * vy
    a = fleet_speed * fleet_speed - v2
    b = -2.0 * rv
    c = -r2

    if abs(a) < _EPS:
        if abs(b) < _EPS:
            return None
        t = -c / b
        if t <= 0.0:
            return None
    else:
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        sd = math.sqrt(disc)
        t1 = (-b - sd) / (2.0 * a)
        t2 = (-b + sd) / (2.0 * a)
        candidates = [t for t in (t1, t2) if t > _EPS]
        if not candidates:
            return None
        t = min(candidates)

    ix = tgt_pos[0] + vx * t
    iy = tgt_pos[1] + vy * t
    return math.atan2(iy - src[1], ix - src[0]), t


def lead_intercept_orbit(
    src: Tuple[float, float],
    current_pos: Tuple[float, float],
    omega: float,
    fleet_speed: float,
    hit_radius: float = 2.0,
    max_turns: int = 500,
    step: int = 1,
) -> Optional[Tuple[float, float]]:
    """
    Intercept a planet circling the sun. `r` is invariant; the planet's
    position at engine turn `step + k` (k turns after launch) is

        theta(k) = atan2(cy - SUN_Y, cx - SUN_X) + k_eff * omega
        pos(k)   = (SUN_X + r * cos(theta(k)),
                    SUN_Y + r * sin(theta(k)))

    where `k_eff = k - 1` when `step == 0` (engine's one-frame freeze:
    positions at step 0 and step 1 are identical) and `k_eff = k`
    otherwise.

    Iterates k = 1..max_turns and returns the k whose radial mismatch
    `|d_k - s * k|` is smallest (within `hit_radius`). Picking the BEST
    k -- not the first below threshold -- matters for small fleets
    where the radial error oscillates across many candidate turns; the
    first crossing isn't necessarily the tightest.

    `hit_radius` should be set to the target planet's actual collision
    radius (1 + ln(production)) so the solver only returns angles the
    fleet can physically land on. The legacy global default of 2.0 was
    looser than even the smallest planets' radius (1.0 for production
    1) and produced "miss by a smidge" outcomes.

    Falls back to the closed-form solver when omega == 0 (a static
    body). Returns None when the target's tangential speed exceeds
    fleet speed at this orbital radius or no integer turn lines up.
    """
    if fleet_speed <= 0.0:
        return None
    if omega == 0.0:
        return lead_intercept(src, current_pos, (0.0, 0.0), fleet_speed)

    cx, cy = current_pos
    r = math.hypot(cx - _SUN_X, cy - _SUN_Y)
    if r < _EPS:
        return None

    theta_now = math.atan2(cy - _SUN_Y, cx - _SUN_X)
    sx, sy = src
    s = fleet_speed
    # One-frame freeze: at step 0 the planet does not advance to step 1.
    freeze_offset = 1 if step == 0 else 0

    best_k = -1
    best_miss = float("inf")
    best_angle = 0.0
    for k in range(1, max_turns + 1):
        k_eff = k - freeze_offset
        theta_k = theta_now + k_eff * omega
        px = _SUN_X + r * math.cos(theta_k)
        py = _SUN_Y + r * math.sin(theta_k)
        d = math.hypot(px - sx, py - sy)
        miss = abs(d - s * k)
        if miss < best_miss:
            best_miss = miss
            best_k = k
            best_angle = math.atan2(py - sy, px - sx)
        elif best_miss <= hit_radius and miss > best_miss + s:
            # Past the tightest crossing on this lap; stop. Future k
            # would only revisit on the next lap (~150+ turns).
            break

    if best_miss <= hit_radius:
        return best_angle, float(best_k)
    return None


def lead_intercept_path(
    src: Tuple[float, float],
    path,
    path_start_idx: int,
    fleet_speed: float,
    hit_radius: float = 2.0,
) -> Optional[Tuple[float, float]]:
    """
    Intercept a target on a known precomputed trajectory (comets).

    For each candidate intercept turn k = 1..K-1:
        d_k = || path[path_start_idx + k] - src ||
        if |d_k - s*k| <= hit_radius
            -> firing at the bearing toward path[path_start_idx + k]
               places the fleet on that point at turn d_k/s ~= k

    Picks the k with the smallest `|d_k - s*k|` (within `hit_radius`),
    not the first crossing -- mirrors the orbit solver's improvement
    so we don't accept a slightly-off k when a tighter one exists
    later in the path.

    Returns the chosen k along with its launch angle, or None when no
    integer turn within the path's remaining horizon admits an
    interception.
    """
    if fleet_speed <= 0.0:
        return None
    horizon = len(path) - int(path_start_idx)
    if horizon < 2:
        return None

    sx, sy = src
    s = fleet_speed
    best_k = -1
    best_miss = float("inf")
    best_angle = 0.0
    for k in range(1, horizon):
        px, py = path[path_start_idx + k]
        d = math.hypot(px - sx, py - sy)
        miss = abs(d - s * k)
        if miss < best_miss:
            best_miss = miss
            best_k = k
            best_angle = math.atan2(py - sy, px - sx)
        elif best_miss <= hit_radius and miss > best_miss + s:
            break

    if best_miss <= hit_radius:
        return best_angle, float(best_k)
    return None
