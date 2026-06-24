# orbit_solvers_torch.py
from __future__ import annotations

from typing import Optional

import torch

SUN_X = 50.0
SUN_Y = 50.0
EPS = 1e-9


@torch.no_grad()
def batched_planet_intercepts(
    src_xy,
    tgt_xy,
    fleet_speed,
    hit_radius,
    is_orbiting,
    *,
    omega: float,
    step: int,
    max_turns: int = 500,
    device: str = "cpu",
):
    """
    Batched intercept solver for static + orbiting planet targets.

    Args:
        src_xy:       [N, 2]
        tgt_xy:       [N, 2]
        fleet_speed:  [N]
        hit_radius:   [N]
        is_orbiting:  [N] bool
        omega:        scalar angular velocity
        step:         current env step
        max_turns:    horizon

    Returns:
        ok:       [N] bool
        angle:    [N] float
        t_hit:    [N] float

    Notes:
        This does not handle comet paths. Keep comet targets on the old
        Python path solver unless/until we batch comet paths separately.
    """
    src_xy = torch.as_tensor(src_xy, dtype=torch.float32, device=device)
    tgt_xy = torch.as_tensor(tgt_xy, dtype=torch.float32, device=device)
    speed = torch.as_tensor(fleet_speed, dtype=torch.float32, device=device)
    radius = torch.as_tensor(hit_radius, dtype=torch.float32, device=device)
    orbiting = torch.as_tensor(is_orbiting, dtype=torch.bool, device=device)

    N = src_xy.shape[0]
    if N == 0:
        return (
            torch.zeros(0, dtype=torch.bool, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
        )

    sx = src_xy[:, 0]
    sy = src_xy[:, 1]
    tx = tgt_xy[:, 0]
    ty = tgt_xy[:, 1]

    ok = torch.zeros(N, dtype=torch.bool, device=device)
    angle = torch.zeros(N, dtype=torch.float32, device=device)
    t_hit = torch.zeros(N, dtype=torch.float32, device=device)

    valid_speed = speed > 0.0

    # ------------------------------------------------------------
    # Static targets: direct center shot.
    # ------------------------------------------------------------
    static_mask = valid_speed & (~orbiting)
    if static_mask.any():
        dx = tx[static_mask] - sx[static_mask]
        dy = ty[static_mask] - sy[static_mask]
        d = torch.sqrt(dx * dx + dy * dy).clamp_min(EPS)

        angle[static_mask] = torch.atan2(dy, dx)
        t_hit[static_mask] = d / speed[static_mask]
        ok[static_mask] = True

    # ------------------------------------------------------------
    # Orbiting targets: vectorized over future integer turns.
    # ------------------------------------------------------------
    orbit_mask = valid_speed & orbiting
    if orbit_mask.any():
        idx = torch.nonzero(orbit_mask, as_tuple=False).flatten()

        sx_o = sx[idx].unsqueeze(1)  # [M, 1]
        sy_o = sy[idx].unsqueeze(1)
        tx_o = tx[idx].unsqueeze(1)
        ty_o = ty[idx].unsqueeze(1)
        speed_o = speed[idx].unsqueeze(1)
        radius_o = radius[idx]

        r = torch.sqrt((tx_o - SUN_X) ** 2 + (ty_o - SUN_Y) ** 2).clamp_min(EPS)
        theta_now = torch.atan2(ty_o - SUN_Y, tx_o - SUN_X)

        k = torch.arange(
            1,
            int(max_turns) + 1,
            dtype=torch.float32,
            device=device,
        ).view(1, -1)  # [1, K]

        freeze_offset = 1.0 if int(step) == 0 else 0.0
        k_eff = k - freeze_offset

        theta = theta_now + k_eff * float(omega)

        px = SUN_X + r * torch.cos(theta)  # [M, K]
        py = SUN_Y + r * torch.sin(theta)

        d = torch.sqrt((px - sx_o) ** 2 + (py - sy_o) ** 2)
        miss = torch.abs(d - speed_o * k)

        best_miss, best_j = torch.min(miss, dim=1)
        good = best_miss <= radius_o

        if good.any():
            good_idx = idx[good]
            bj = best_j[good]

            px_best = px[good, :].gather(1, bj.view(-1, 1)).squeeze(1)
            py_best = py[good, :].gather(1, bj.view(-1, 1)).squeeze(1)

            angle[good_idx] = torch.atan2(py_best - sy[good_idx], px_best - sx[good_idx])
            t_hit[good_idx] = bj.to(torch.float32) + 1.0
            ok[good_idx] = True

    return ok, angle, t_hit