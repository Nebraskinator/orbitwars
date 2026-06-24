"""
Shared constants and primitives for Orbit Wars observation encoding.

Two-hot encoding maps a continuous scalar onto a `n_bins`-length vector
where the two bins flanking the value share weight (linear interp). This
gives the network a soft, position-aware representation that's easier to
attend over than a raw float, while remaining differentiable end-to-end.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

N_BINS = 51                 # two-hot resolution for positions, step number, etc.
# Ship/garrison counts use a coarser two-hot. Most decisions only care about
# the order of magnitude of a fleet ("a few", "tens", "hundreds"), not its
# exact value, so 9 log-spaced bins cover the useful range without inflating
# the token width.
N_SHIP_BINS = 9
# Cap on planet/comet tokens (shared encoder). Comets live inline in the
# engine's planets list, so this cap covers both: 32 planets + 4 comets per
# spawn group = 36 total bodies. The pointer-head action space therefore has
# exactly MAX_PLANETS target buckets.
MAX_PLANETS = 44
MAX_FLEETS = 128            # cap on fleet tokens; over-cap fleets dropped by ship count
MAX_PLAYERS = 4             # FFA cap; 1v1 leaves opp slots zero

# ---------------------------------------------------------------------------
# Two-hot ranges (chosen with a small margin past observed extremes)
# ---------------------------------------------------------------------------

POS_LOW, POS_HIGH = 0.0, 100.0                  # board coordinates
PLANET_VEL_LOW, PLANET_VEL_HIGH = -5.0, 5.0     # planet/comet dx,dy bounds
FLEET_VEL_LOW, FLEET_VEL_HIGH = -6.5, 6.5       # shipSpeed cap = 6 + headroom
LOG_SHIPS_LOW, LOG_SHIPS_HIGH = 0.0, 12.0       # ln(1+ships); covers ~163K

# ---------------------------------------------------------------------------
# Game constants (mirror Kaggle defaults)
# ---------------------------------------------------------------------------

BOARD_SIZE = 100.0
SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS = 10.0
ORBITAL_LIMIT = 50.0     # planets with r_sun + radius < ORBITAL_LIMIT orbit
SHIP_SPEED_MAX = 6.0
COMET_SPEED = 4.0
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
#
# Per-source-planet factored multi-discrete action: (target_idx, option_idx).
#
# target_idx picks one of MAX_PLANETS body slots.
#
# option_idx is now intentionally binary:
#
#   OPT_SKIP     = 0   no fleet launched. Canonical label pairs this with
#                      target_idx == source slot.
#
#   OPT_MAX_SEND = 1   launch the full current source-planet garrison toward
#                      target_idx. The interpreter resolves the firing angle,
#                      but it does not choose a fractional or takeover-sized
#                      ship count.
#
# There are no percentage buckets, no takeover option, and no fraction helpers.

OPT_SKIP = 0
OPT_MAX_SEND = 1

N_ACTION_OPTIONS = 2
# ---------------------------------------------------------------------------
# Two-hot helpers
# ---------------------------------------------------------------------------

def two_hot_inplace(
    buf: np.ndarray,
    offset: int,
    value: float,
    low: float,
    high: float,
    n_bins: int = N_BINS,
) -> None:
    """
    Write a 2-hot encoding of `value` into buf[offset:offset+n_bins].

    Assumes that range of buf is already zero. Out-of-range values are
    clipped onto the edge bin.
    """
    if value <= low:
        buf[offset] = 1.0
        return
    if value >= high:
        buf[offset + n_bins - 1] = 1.0
        return
    pos = (value - low) / (high - low) * (n_bins - 1)
    lo_bin = int(pos)              # floor for non-negative pos
    w_hi = pos - lo_bin
    buf[offset + lo_bin] = 1.0 - w_hi
    if lo_bin + 1 < n_bins:
        buf[offset + lo_bin + 1] = w_hi


def two_hot(value, low, high, n_bins=N_BINS):
    out = np.zeros(n_bins, dtype=np.float32)
    two_hot_inplace(out, 0, value, low, high, n_bins)
    return out


# ---------------------------------------------------------------------------
# Game math
# ---------------------------------------------------------------------------

_LOG_1000 = math.log(1000.0)
_EPS = 1e-9


def segment_enters_circle(p0, p1, c0, c1, R):
    """
    Smallest t in [0, 1] where a moving point segment p0->p1 enters
    a moving disc c0->c1 with radius R. Returns None on miss.

    This mirrors the validated reference simulator's collision primitive.
    """
    vx = (p1[0] - p0[0]) - (c1[0] - c0[0])
    vy = (p1[1] - p0[1]) - (c1[1] - c0[1])
    rx = p0[0] - c0[0]
    ry = p0[1] - c0[1]

    a = vx * vx + vy * vy
    c = rx * rx + ry * ry - R * R

    if c <= 0.0:
        return 0.0

    if a < _EPS:
        return None

    b = 2.0 * (vx * rx + vy * ry)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    if t2 < 0.0 or t1 > 1.0:
        return None

    return max(0.0, t1)


def fleet_speed(ships):
    """Match the engine's logarithmic speed-vs-size curve."""
    if ships <= 0:
        return 0.0
    if ships >= 1000:
        return SHIP_SPEED_MAX
    s = math.log(ships) / _LOG_1000
    return 1.0 + (SHIP_SPEED_MAX - 1.0) * (s ** 1.5)


def is_orbiting(r_from_sun, planet_radius):
    """A planet orbits when its outer edge stays inside the orbital ring."""
    return (r_from_sun + planet_radius) < ORBITAL_LIMIT


def orbit_position(theta0, r, step, omega):
    """
    Position of an orbiting planet at engine step `step`.

    The engine has a one-frame freeze at game start: planet positions at
    step 0 and step 1 are identical, then rotation accumulates from step 2.
    Equivalent formula: theta(t) = theta0 + max(0, t-1) * omega.
    """
    eff_t = max(0, step - 1)
    theta = theta0 + eff_t * omega
    return SUN_X + r * math.cos(theta), SUN_Y + r * math.sin(theta)
