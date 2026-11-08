"""Rough minimum-lap-time estimate: optimize the racing line and speed profile, same car limits.

    python -m sdc.racer.min_lap 10000 10001 ...

The line is a smooth lateral offset from the centerline (40-harmonic Fourier series, kept within
the road), optimized by gradient descent on lap time. Speed profile for a given line: the fastest
that obeys the cornering limit (A_LAT), acceleration (ACCEL) and braking (BRAKE), from a standing
start on the start line, one lap. Continuous time, so it ignores the sim's 0.1 s steps; the
optimizer can also stop at a local optimum. Treat the result as an estimate good to a few percent.
Prints: track, best centerline-only lap (s), optimized-line lap (s).
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from sdc.racer import sim
from sdc.racer import track as trk


def lap_time(t: trk.Track, offset: torch.Tensor) -> torch.Tensor:
    p = torch.tensor(t.center) + offset[:, None] * torch.tensor(t.normal)
    seg = torch.roll(p, -1, 0) - p
    ds = seg.norm(dim=1)
    h = torch.atan2(seg[:, 1], seg[:, 0])
    dh = torch.remainder(h - torch.roll(h, 1) + np.pi, 2 * np.pi) - np.pi
    kappa = dh.abs() / (0.5 * (ds + torch.roll(ds, 1)))
    v_corner = torch.clamp(torch.sqrt(sim.A_LAT / (kappa + 1e-6)), max=sim.V_MAX)
    n = len(ds)
    v = [torch.tensor(0.0)]
    for i in range(1, n + 1):  # accelerate from standstill
        v.append(torch.minimum(v_corner[i % n], torch.sqrt(v[-1] ** 2 + 2 * sim.ACCEL * ds[i - 1])))
    for i in range(n - 1, -1, -1):  # brake in time for every corner
        v[i] = torch.minimum(v[i], torch.sqrt(v[i + 1] ** 2 + 2 * sim.BRAKE * ds[i]))
    v = torch.stack(v)
    return (ds / torch.clamp(0.5 * (v[:-1] + v[1:]), min=0.3)).sum()


def estimate(seed: int, harmonics: int = 40, iters: int = 500) -> tuple[float, float]:
    t = trk.generate(seed)
    n = trk.N_POINTS
    th = torch.arange(n) * 2 * np.pi / n
    basis = torch.stack([f(k * th) for k in range(1, harmonics + 1) for f in (torch.cos, torch.sin)], 1)
    lim = sim.OFFTRACK_LIMIT - 0.1
    ramp = torch.clamp(torch.arange(n) / 5.0, max=1.0)  # start on the centerline, like the sim
    w = torch.zeros(basis.shape[1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.02)
    base = float(lap_time(t, torch.zeros(n)))
    best = base
    for _ in range(iters):
        loss = lap_time(t, lim * torch.tanh(basis @ w) * ramp)
        opt.zero_grad()
        loss.backward()
        opt.step()
        best = min(best, loss.item())
    return base, best


if __name__ == "__main__":
    torch.set_num_threads(1)
    for s in (int(x) for x in sys.argv[1:]):
        print(s, *(round(x, 2) for x in estimate(s)), flush=True)
