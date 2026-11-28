"""A worn dirt-track surface on a loop: a groove of packed dirt along a racing line, loose dirt off it, and bumps
where cars brake and accelerate (docs/phases/surface-plan.md, realism review docs/research/surface-plan-review.md).

Physics switches (sdc/jumps/real.py):
  groove   a per-tyre grip multiplier from a grip map rasterised on the heightfield grid: higher in the groove,
           lower off it, lowest in the marbles on the outside of corner exits
  bbumps   braking bumps where the race line brakes hard, acceleration bumps on corner exits, a touchdown dish
           after each jump's landing; added to the heightfield at its grid nodes (not splatted)

Each loop gets a track state tau from loam (0) to blue groove (1), drawn from its own seed, so a loop always has the
same surface. Loam: a small grip contrast and big bumps. Blue groove: a big grip contrast and small bumps.
The race line: a minimum-curvature line within +-LINE_MAX of the centreline, apexing LATE_APEX later, plus a slow
wander (drivers don't all take the same line).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sdc.jumps import loop as lp

LINE_MAX = 1.0            # m, the race line stays this far inside the centreline (0.5 m from the banks' feet)
LINE_DS = 0.5             # m, spacing of the race-line solve
LATE_APEX = 0.75          # m, the line's apex moved later than minimum curvature (drivers apex late)
WANDER = (0.15, 25.0)     # (RMS m, wavelength m) of the line's wander (long: it mustn't add corners)
GROOVE_CORE, GROOVE_EDGE = 0.4, 0.8   # m from the line: full groove inside the core, none beyond the edge
CORNER_WIDEN = 0.3        # m extra groove half-width in tight corners
GRIP_CLIP = (0.65, 1.18)  # the map's multiplier range
# race-line speed profile (for bump placement only): lateral, braking and drive limits, top speed
A_LAT, A_BRK, A_ACC, V_TOP = 7.0, 6.0, 4.0, 12.0
BUMP_BAND = 0.5           # m half-width of the bumpy band around the line (tapered)
BUMP_PP = (0.8, 2.0)      # cm, peak-to-peak bump height at the hardest braking: blue groove .. loam
DISH = (0.008, 0.02, 1.5)   # touchdown dish: depth range m (blue groove .. loam), length m


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _circ_blur(v, sigma_pts):
    return lp._blur(v, sigma_pts)


def race_line(c, rng):
    """Lateral offset d*(s) of the race line at every centreline sample (n,)."""
    n = len(c.center)
    step = max(1, int(round(LINE_DS / c.ds)))
    idx = np.arange(0, n, step)
    m = len(idx)
    C = c.center[idx]
    th = c.heading[idx]
    N = np.stack([-np.sin(th), np.cos(th)], 1)
    # minimise the line's bending energy sum(kappa^2 ds) within the bounds (L-BFGS on the true curvature,
    # the offset squashed into +-LINE_MAX by tanh)
    import torch
    Ct, Nt = torch.tensor(C), torch.tensor(N)
    z = torch.zeros(m, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([z], lr=1, max_iter=500, history_size=50, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-12)

    def energy():
        opt.zero_grad()
        P = Ct + (LINE_MAX * torch.tanh(z))[:, None] * Nt
        a_, b_ = P - torch.roll(P, 1, 0), torch.roll(P, -1, 0) - P
        ang = torch.atan2(a_[:, 0] * b_[:, 1] - a_[:, 1] * b_[:, 0], (a_ * b_).sum(1))
        loss = (ang ** 2 / (0.5 * (a_.norm(dim=1) + b_.norm(dim=1)))).sum()
        loss.backward()
        return loss

    for _ in range(3):
        opt.step(energy)
    d = (LINE_MAX * torch.tanh(z)).detach().numpy()
    # back to every sample, apex later, then a slow wander
    q = np.arange(n) / step
    dl = np.interp(q, np.arange(m + 1), np.append(d, d[0]))
    dl = np.roll(dl, int(round(LATE_APEX / c.ds)))
    rms, wl = WANDER
    s = np.arange(n) * c.ds
    wander = np.zeros(n)
    for _ in range(4):   # a few waves whose wavelengths fit the lap
        k = max(1, int(round(c.length / (wl * rng.uniform(0.7, 1.4)))))
        wander += np.sin(2 * np.pi * k * s / c.length + rng.uniform(0, 2 * np.pi))
    wander *= rms / max(wander.std(), 1e-9)
    return np.clip(_circ_blur(dl, 1.0 / c.ds) + wander, -LINE_MAX - 0.1, LINE_MAX + 0.1)


def speed_profile(c, dline):
    """Race-line speed (n,) from its curvature with braking and drive limits (forward-backward passes)."""
    n = len(c.center)
    th = c.heading
    N = np.stack([-np.sin(th), np.cos(th)], 1)
    P = c.center + dline[:, None] * N
    seg = np.roll(P, -1, 0) - P
    hd = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    L = np.maximum(np.hypot(seg[:, 0], seg[:, 1]), 1e-6)
    kap = _circ_blur(np.gradient(hd) / L, 1.5 / c.ds)   # the line is solved 0.5 m apart: smooth its kinks
    v = np.minimum(V_TOP, np.sqrt(A_LAT / np.maximum(np.abs(kap), 1e-6)))
    for _ in range(2):   # twice round, so the lap's seam is consistent
        for i in range(n):                       # drive limit going forward
            j = (i - 1) % n
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * A_ACC * L[j]))
        for i in range(n - 1, -1, -1):           # braking limit going backward
            j = (i + 1) % n
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * A_BRK * L[i]))
    dv = (np.roll(v, -1) - np.roll(v, 1)) / (2 * c.ds)
    brake = np.clip(-v * dv / A_BRK, 0, 1)
    accel = np.clip(v * dv / A_ACC, 0, 1)
    return v, brake, accel, kap


@dataclass
class Surface:
    tau: float
    dline: np.ndarray       # (n,) race-line offset
    speed: np.ndarray       # (n,) race-line speed
    grip: np.ndarray        # (N, N) grip multiplier on the hfield grid (1 off the strip)
    bump: np.ndarray        # (N, N) height added, m (0 off the strip)

    def grip_at(self, X, Y):
        ix = np.clip(np.rint((np.asarray(X) + lp.BOX / 2) / lp.D).astype(int), 0, lp.N - 1)
        iy = np.clip(np.rint((np.asarray(Y) + lp.BOX / 2) / lp.D).astype(int), 0, lp.N - 1)
        return self.grip[iy, ix]


def _track_coords(c):
    """For every hfield node near the lane: its flat index, centreline index, s and d (exact local projection)."""
    n = len(c.center)
    nx, ny = -np.sin(c.heading), np.cos(c.heading)
    near = np.full((lp.N, lp.N), -1, int)
    best = np.full((lp.N, lp.N), np.inf)
    r = int(np.ceil(lp.STRIP / lp.D)) + 1
    off = np.arange(-r, r + 1)
    for i in range(0, n, 2):   # each centreline sample claims the grid nodes around it that it is closest to
        cx, cy = c.center[i]
        ix = int(np.rint((cx + lp.BOX / 2) / lp.D)) + off
        iy = int(np.rint((cy + lp.BOX / 2) / lp.D)) + off
        ix, iy = ix[(ix >= 0) & (ix < lp.N)], iy[(iy >= 0) & (iy < lp.N)]
        X = ix * lp.D - lp.BOX / 2
        Y = iy * lp.D - lp.BOX / 2
        dist = (X[None, :] - cx) ** 2 + (Y[:, None] - cy) ** 2
        sub = best[np.ix_(iy, ix)]
        win = dist < sub
        best[np.ix_(iy, ix)] = np.where(win, dist, sub)
        cur = near[np.ix_(iy, ix)]
        near[np.ix_(iy, ix)] = np.where(win, i, cur)
    flat = np.flatnonzero((near >= 0) & (best < (lp.STRIP + 0.2) ** 2))
    iy, ix = np.divmod(flat, lp.N)
    i = near.ravel()[flat]
    X, Y = ix * lp.D - lp.BOX / 2, iy * lp.D - lp.BOX / 2
    rx, ry = X - c.center[i, 0], Y - c.center[i, 1]
    tx, ty = np.cos(c.heading[i]), np.sin(c.heading[i])
    along = rx * tx + ry * ty
    d = rx * nx[i] + ry * ny[i]
    s = (i * c.ds + along) % c.length
    return flat, i, s, d


def build(c) -> Surface:
    """The worn surface of loop c (deterministic in its seed)."""
    rng = np.random.default_rng([int(c.seed), 47])
    tau = float(rng.uniform(0.0, 1.0))
    dline = race_line(c, rng)
    v, brake, accel, kap = speed_profile(c, dline)
    flat, i, s, d = _track_coords(c)
    q = s / c.ds
    i0 = np.floor(q).astype(int) % len(c.center)
    f = q - np.floor(q)
    lerp = lambda a: a[i0] * (1 - f) + a[(i0 + 1) % len(a)] * f
    dl = lerp(dline)
    e = d - dl
    # --- grip ---
    gain, loss, marb = 0.03 + 0.12 * tau, 0.03 + 0.12 * tau, 0.10 + 0.20 * tau
    tight = _smooth(np.abs(lerp(kap)) / 0.25)
    core = GROOVE_CORE + CORNER_WIDEN * tight
    edge = GROOVE_EDGE + CORNER_WIDEN * tight
    w = 1.0 - _smooth((np.abs(e) - core) / (edge - core))          # 1 in the groove, 0 off it, smooth between
    # marbles: outside of the line, on corner exits (the corner weight carried 3 m past the corner, blurred)
    n = len(c.center)
    cw = _smooth((np.abs(c.kappa) - 0.05) / 0.15)                     # the corners (centreline curvature)
    shift = int(round(3.0 / c.ds))
    exitw = np.maximum.reduce([np.roll(cw, k) for k in range(0, shift + 1, max(1, shift // 6))])
    exitw = _circ_blur(exitw, 1.0 / c.ds)
    side = np.sign(lerp(c.kappa))                                        # + = corner centre to the left
    outside = _smooth((-e * side - 0.3) / 0.5)                         # smooth ramp outward of the line
    mb = lerp(exitw) * outside * (1 - w)
    g = 1.0 + gain * w - loss * (1 - w) - marb * mb
    grip = np.ones(lp.N * lp.N)
    grip[flat] = np.clip(g, *GRIP_CLIP)
    # --- bumps ---
    clear = np.ones(n, bool)
    x = np.arange(n) * c.ds
    for j in c.jumps:
        clear &= ~((x > j.start - 2.0) & (x < j.end + 1.0))
    inten = np.maximum(np.where(brake > 0.3, brake, 0.0), 0.6 * np.where(accel > 0.4, accel, 0.0)) * clear
    inten = _circ_blur(inten, 0.5 / c.ds)
    lam = np.clip(0.045 * v, 0.25, 0.55)
    phase = np.cumsum(2 * np.pi / lam * c.ds)
    # each bump zone gets its own ridge skew (up to +-15 deg) and a phase offset
    zone = np.cumsum(np.diff(np.r_[0, (inten > 0.05).astype(int)]) == 1)
    zs = rng.uniform(-15, 15, zone.max() + 2)
    zp = rng.uniform(0, 2 * np.pi, zone.max() + 2)
    pp = (BUMP_PP[1] - (BUMP_PP[1] - BUMP_PP[0]) * tau) / 100.0       # m peak-to-peak at full intensity
    band = np.cos(np.clip(e / BUMP_BAND, -1, 1) * np.pi / 2) ** 2
    lam_n = lerp(lam)
    ph = lerp(phase) + zp[zone[i0]] + 2 * np.pi / lam_n * np.tan(np.deg2rad(zs[zone[i0]])) * e
    h = 0.5 * pp * lerp(inten) * band * np.sin(ph)
    # touchdown dish after each landing: a smooth hollow on the line where cars land
    depth = DISH[1] - (DISH[1] - DISH[0]) * tau
    for j in c.jumps:
        s0 = j.land_start if j.kind == "drop" else 0.5 * (j.land_start + j.end)
        ds_ = (s - (s0 + 0.5 * DISH[2])) / (0.5 * DISH[2])
        h -= depth * np.where(np.abs(ds_) < 1, np.cos(ds_ * np.pi / 2) ** 2, 0.0) * band
    bump = np.zeros(lp.N * lp.N)
    bump[flat] = h
    return Surface(tau, dline, v, grip.reshape(lp.N, lp.N), bump.reshape(lp.N, lp.N))


_CACHE: dict = {}


def get(c) -> Surface:
    key = (int(c.seed), len(c.center))
    if key not in _CACHE:
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[key] = build(c)
    return _CACHE[key]
