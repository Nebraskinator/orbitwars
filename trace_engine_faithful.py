"""
Engine-faithful fleet first-hit trace (two-phase collision), built to match the
real orbit_wars engine bit-for-bit (validated end-to-end by test_engine_oracle.py
at rotation offset -1, 1054/1054).

This module holds, on the SAME interface as orbit_wars_numpy.trace_fleets_first_hit_np:
  * engine_trace_loop_ref : slow per-tick reference (the validated loop)
  * trace_first_hit_v2     : vectorized implementation

and a __main__ self-test asserting v2 == loop_ref bitwise on random geometry.
Once green, this logic replaces trace_fleets_first_hit_np / _torch.

Engine semantics reproduced (orbit_src/orbit_wars/orbit_wars.py):
  Per tick, for a fleet moving F_t -> F_{t+1} (straight line, constant speed):
    Phase A: test the fleet SEGMENT against each body at its STATIC start
             position P_t = body_pos[t]; hit if point_to_segment_distance < r.
             First body in array order wins.
    then OOB on the endpoint F_{t+1}; then SUN (segment vs static sun).
    Phase B: each body SWEEPS P_t -> P_{t+1} = body_pos[t+1] and catches the
             fleet at its POINT F_{t+1}; hit if point_to_segment_distance < r.
             First body in array order wins.
  Priority: Phase-A planet > OOB > sun > Phase-B planet.
  body_pos[t] already encodes the engine's rotation lag (eff_t = base_step-1),
  so body_pos[t]=P_t and body_pos[t+1]=P_{t+1}.

All math in float64 to match the engine (Python floats).
"""

from __future__ import annotations

import numpy as np

SUN_X = 50.0
SUN_Y = 50.0
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0


# --------------------------------------------------------------------------
# Scalar helpers (engine-exact).
# --------------------------------------------------------------------------
def _pt_seg_d2(px, py, ax, ay, bx, by):
    """Squared min distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    abx = bx - ax
    aby = by - ay
    l2 = abx * abx + aby * aby
    if l2 == 0.0:
        dx, dy = px - ax, py - ay
        return dx * dx + dy * dy
    t = ((px - ax) * abx + (py - ay) * aby) / l2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = ax + t * abx
    cy = ay + t * aby
    return (px - cx) ** 2 + (py - cy) ** 2


def _seg_circle_entry(p0x, p0y, p1x, p1y, cx, cy, r):
    """First s in [0,1] where segment p0->p1 enters disc(c,r). Assumes a hit
    exists (min-dist < r). Returns s in [0,1] (0 if p0 already inside)."""
    dx, dy = p1x - p0x, p1y - p0y
    fx, fy = p0x - cx, p0y - cy
    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    if c <= 0.0:
        return 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        disc = 0.0
    s = (-b - np.sqrt(disc)) / (2.0 * a) if a > 0.0 else 0.0
    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0
    return s


# --------------------------------------------------------------------------
# Reference loop (validated engine-faithful).
# --------------------------------------------------------------------------
def engine_trace_loop_ref(pos, angles, speeds, body_pids, body_radii,
                          body_pos, body_active, max_turns):
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2)
    angles = np.asarray(angles, dtype=np.float64).reshape(-1)
    speeds = np.asarray(speeds, dtype=np.float64).reshape(-1)
    body_pids = np.asarray(body_pids).reshape(-1)
    body_radii = np.asarray(body_radii, dtype=np.float64).reshape(-1)
    body_pos = np.asarray(body_pos, dtype=np.float64)
    body_active = np.asarray(body_active, dtype=bool)

    F = pos.shape[0]
    B = body_pids.shape[0]
    T = max(0, int(max_turns))
    target = np.full(F, -1, dtype=np.int64)
    eta = np.full(F, np.nan, dtype=np.float64)
    blocked = np.ones(F, dtype=bool)
    if F == 0 or T == 0:
        return target, eta, blocked

    for f in range(F):
        if speeds[f] <= 0.0:
            continue
        vx = speeds[f] * np.cos(angles[f])
        vy = speeds[f] * np.sin(angles[f])
        fx, fy = float(pos[f, 0]), float(pos[f, 1])
        for t in range(T):
            oldx, oldy = fx, fy
            newx, newy = fx + vx, fy + vy
            # Phase A: fleet segment vs static body at P_t.
            hitb = -1
            for b in range(B):
                if not body_active[t, b]:
                    continue
                r = body_radii[b]
                if _pt_seg_d2(body_pos[t, b, 0], body_pos[t, b, 1],
                              oldx, oldy, newx, newy) < r * r:
                    hitb = b
                    break
            if hitb >= 0:
                target[f] = body_pids[hitb]
                s = _seg_circle_entry(oldx, oldy, newx, newy,
                                      body_pos[t, hitb, 0], body_pos[t, hitb, 1],
                                      body_radii[hitb])
                eta[f] = t + s
                blocked[f] = False
                break
            # OOB
            if not (0.0 <= newx <= BOARD_SIZE and 0.0 <= newy <= BOARD_SIZE):
                break
            # SUN
            if _pt_seg_d2(SUN_X, SUN_Y, oldx, oldy, newx, newy) < SUN_RADIUS * SUN_RADIUS:
                break
            # Phase B: body sweep P_t->P_{t+1} vs fleet point F_{t+1}.
            hitb = -1
            for b in range(B):
                if not (body_active[t, b] and body_active[t + 1, b]):
                    continue
                r = body_radii[b]
                if _pt_seg_d2(newx, newy,
                              body_pos[t, b, 0], body_pos[t, b, 1],
                              body_pos[t + 1, b, 0], body_pos[t + 1, b, 1]) < r * r:
                    hitb = b
                    break
            if hitb >= 0:
                target[f] = body_pids[hitb]
                eta[f] = t + 1.0
                blocked[f] = False
                break
            fx, fy = newx, newy
    return target, eta, blocked


# --------------------------------------------------------------------------
# Vectorized implementation.
# --------------------------------------------------------------------------
def _pt_seg_d2_vec(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    l2 = abx * abx + aby * aby
    safe = np.where(l2 > 0.0, l2, 1.0)
    t = ((px - ax) * abx + (py - ay) * aby) / safe
    t = np.clip(t, 0.0, 1.0)
    t = np.where(l2 > 0.0, t, 0.0)
    cx = ax + t * abx
    cy = ay + t * aby
    return (px - cx) ** 2 + (py - cy) ** 2


def trace_first_hit_v2(pos, angles, speeds, body_pids, body_radii,
                       body_pos, body_active, max_turns):
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2)
    angles = np.asarray(angles, dtype=np.float64).reshape(-1)
    speeds = np.asarray(speeds, dtype=np.float64).reshape(-1)
    body_pids = np.asarray(body_pids).reshape(-1).astype(np.int64)
    body_radii = np.asarray(body_radii, dtype=np.float64).reshape(-1)
    body_pos = np.asarray(body_pos, dtype=np.float64)
    body_active = np.asarray(body_active, dtype=bool)

    F = pos.shape[0]
    B = body_pids.shape[0]
    T = max(0, int(max_turns))
    target = np.full(F, -1, dtype=np.int64)
    eta = np.full(F, np.nan, dtype=np.float64)
    blocked = np.ones(F, dtype=bool)
    if F == 0 or T == 0 or B == 0:
        return target, eta, blocked

    active0 = speeds > 0.0
    vx = speeds * np.cos(angles)
    vy = speeds * np.sin(angles)
    v = np.stack([vx, vy], axis=-1)                      # [F,2]

    traj = np.empty((T + 1, F, 2), dtype=np.float64)
    traj[0] = pos
    for t in range(T):
        traj[t + 1] = traj[t] + v
    p0 = traj[:-1]                                        # [T,F,2]
    p1 = traj[1:]
    c0 = body_pos[:T]                                     # [T,B,2]
    c1 = body_pos[1:T + 1]
    rb2 = body_radii * body_radii                        # [B]
    bactA = body_active[:T]                              # [T,B]
    bactB = body_active[:T] & body_active[1:T + 1]

    # --- single AABB broadphase + sparse two-phase exact kernel ---
    # The fleet SEGMENT AABB vs the SWEPT-body AABB (min/max over c0,c1, +/- r)
    # is a superset of both Phase-A (static body) and Phase-B (swept body)
    # candidates, so one nonzero feeds both exact tests. We never build the
    # dense [T,F,B] float distance tensor.
    fminx = np.minimum(p0[..., 0], p1[..., 0]); fmaxx = np.maximum(p0[..., 0], p1[..., 0])
    fminy = np.minimum(p0[..., 1], p1[..., 1]); fmaxy = np.maximum(p0[..., 1], p1[..., 1])
    sbminx = np.minimum(c0[..., 0], c1[..., 0]) - body_radii
    sbmaxx = np.maximum(c0[..., 0], c1[..., 0]) + body_radii
    sbminy = np.minimum(c0[..., 1], c1[..., 1]) - body_radii
    sbmaxy = np.maximum(c0[..., 1], c1[..., 1]) + body_radii
    ov = ((fmaxx[:, :, None] >= sbminx[:, None, :]) & (fminx[:, :, None] <= sbmaxx[:, None, :])
          & (fmaxy[:, :, None] >= sbminy[:, None, :]) & (fminy[:, :, None] <= sbmaxy[:, None, :])
          & bactA[:, None, :])                           # [T,F,B] bool (bactB <= bactA)
    A_hit = np.zeros((T, F, B), dtype=bool)
    B_hit = np.zeros((T, F, B), dtype=bool)
    it, jf, kb = np.nonzero(ov)
    if it.size:
        # Phase A: static body(c0) vs fleet segment.
        d2a = _pt_seg_d2_vec(c0[it, kb, 0], c0[it, kb, 1],
                             p0[it, jf, 0], p0[it, jf, 1], p1[it, jf, 0], p1[it, jf, 1])
        A_hit[it, jf, kb] = d2a < rb2[kb]
        # Phase B: fleet endpoint vs swept body(c0->c1); extra bactB gate.
        d2b = _pt_seg_d2_vec(p1[it, jf, 0], p1[it, jf, 1],
                             c0[it, kb, 0], c0[it, kb, 1], c1[it, kb, 0], c1[it, kb, 1])
        B_hit[it, jf, kb] = (d2b < rb2[kb]) & bactB[it, kb]
    A_any = A_hit.any(-1)                                 # [T,F]
    A_fb = np.argmax(A_hit, axis=-1)                      # [T,F] first body
    B_any = B_hit.any(-1)
    B_fb = np.argmax(B_hit, axis=-1)

    oob = ((p1[..., 0] < 0.0) | (p1[..., 0] > BOARD_SIZE)
           | (p1[..., 1] < 0.0) | (p1[..., 1] > BOARD_SIZE))   # [T,F]
    d2sun = _pt_seg_d2_vec(SUN_X, SUN_Y, p0[..., 0], p0[..., 1],
                           p1[..., 0], p1[..., 1])             # [T,F]
    sun = d2sun < (SUN_RADIUS * SUN_RADIUS)

    terminal = (A_any | oob | sun | B_any) & active0[None, :]  # [T,F]
    term_any = terminal.any(0)
    t_star = np.argmax(terminal, axis=0)                       # [F]
    cols = np.arange(F)

    A_s = A_any[t_star, cols]
    oob_s = oob[t_star, cols]
    sun_s = sun[t_star, cols]
    B_s = B_any[t_star, cols]

    cause_A = A_s & term_any
    cause_B = (~A_s) & (~oob_s) & (~sun_s) & B_s & term_any

    # Phase-A entry time at the resolved (t_star, first body).
    if cause_A.any():
        fbA = A_fb[t_star, cols]                              # [F]
        cAx = c0[t_star, fbA, 0]
        cAy = c0[t_star, fbA, 1]
        rA = body_radii[fbA]
        p0x = p0[t_star, cols, 0]; p0y = p0[t_star, cols, 1]
        p1x = p1[t_star, cols, 0]; p1y = p1[t_star, cols, 1]
        dx = p1x - p0x; dy = p1y - p0y
        fx = p0x - cAx; fy = p0y - cAy
        a = dx * dx + dy * dy
        b = 2.0 * (fx * dx + fy * dy)
        cc = fx * fx + fy * fy - rA * rA
        disc = np.maximum(b * b - 4.0 * a * cc, 0.0)
        s = np.where(cc <= 0.0, 0.0,
                     (-b - np.sqrt(disc)) / np.where(a > 0.0, 2.0 * a, 1.0))
        s = np.clip(s, 0.0, 1.0)
        idx = np.where(cause_A)[0]
        target[idx] = body_pids[fbA[idx]]
        eta[idx] = t_star[idx] + s[idx]
        blocked[idx] = False

    if cause_B.any():
        fbB = B_fb[t_star, cols]
        idx = np.where(cause_B)[0]
        target[idx] = body_pids[fbB[idx]]
        eta[idx] = t_star[idx] + 1.0
        blocked[idx] = False

    return target, eta, blocked


# --------------------------------------------------------------------------
# Self-test: v2 == loop_ref bitwise on random geometry.
# --------------------------------------------------------------------------
def _random_case(rng, F, B, T):
    pos = rng.uniform(5, 95, size=(F, 2))
    angles = rng.uniform(0, 2 * np.pi, size=F)
    ships = rng.integers(1, 200, size=F).astype(np.float64)
    speeds = 1.0 + 5.0 * (np.log(np.clip(ships, 1, None)) / np.log(1000)) ** 1.5
    speeds = np.minimum(speeds, 6.0)
    body_pids = np.arange(B, dtype=np.int64)
    body_radii = rng.uniform(1.0, 2.5, size=B)
    # Body trajectories: some static, some rotating about the sun.
    theta0 = rng.uniform(0, 2 * np.pi, size=B)
    r_orb = rng.uniform(12, 45, size=B)
    omega = rng.uniform(0.025, 0.05)
    rotating = rng.random(B) < 0.5
    body_pos = np.empty((T + 1, B, 2))
    for t in range(T + 1):
        ang = theta0 + omega * t
        x = np.where(rotating, 50 + r_orb * np.cos(ang), 50 + r_orb * np.cos(theta0))
        y = np.where(rotating, 50 + r_orb * np.sin(ang), 50 + r_orb * np.sin(theta0))
        body_pos[t, :, 0] = x
        body_pos[t, :, 1] = y
    body_active = np.ones((T + 1, B), dtype=bool)
    return pos, angles, speeds, body_pids, body_radii, body_pos, body_active


def main():
    rng = np.random.default_rng(0)
    T = 30
    n_tgt = n_eta = n_blk = 0
    trials = 200
    worst_eta = 0.0
    for it in range(trials):
        F = int(rng.integers(1, 40))
        B = int(rng.integers(1, 30))
        case = _random_case(rng, F, B, T)
        t1, e1, b1 = engine_trace_loop_ref(*case, T)
        t2, e2, b2 = trace_first_hit_v2(*case, T)
        n_tgt += int((t1 != t2).sum())
        n_blk += int((b1 != b2).sum())
        em = ~(np.isnan(e1) & np.isnan(e2))
        d = np.abs(np.nan_to_num(e1) - np.nan_to_num(e2))[em]
        if d.size:
            worst_eta = max(worst_eta, float(d.max()))
        n_eta += int((d > 1e-9).sum())
    print(f"trials={trials}  target mism={n_tgt}  blocked mism={n_blk}  "
          f"eta mism(>1e-9)={n_eta}  worst|Δeta|={worst_eta:.2e}")
    print("PASS" if (n_tgt == 0 and n_blk == 0 and n_eta == 0) else "FAIL")


if __name__ == "__main__":
    main()
