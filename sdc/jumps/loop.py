"""Phase 4.6: closed-loop courses with turns, laid on a 2D heightfield.

A loop is a random polygon (4-6 corners) with its corners rounded off (a Gaussian blur of the outline), so it has true
straights joined by turns of different radii. A spare straight gets a chicane (an S-bend), so most loops turn both ways. It is scaled to a lap of LAP metres. The lane is the
same as the jump courses': DRIVE_W (3 m) between low banks.

On the lane:
  jumps        Phase 4.5's roller-course jumps (double, table, drop), one on each of three straights; the same
               generator and design speeds as the straight courses (course.build_jump).
  turns        some flat, some banked (the inside lower, up to BANK_MAX degrees), drawn per corner.
  whoops       one stretch of small rolling bumps.
  rough ground a few cm of smooth roughness everywhere except on the jumps.

Coordinates. Everything that scores or teaches reads the car's position in track coordinates: s = metres along the
centreline from the start/finish line (0..length), d = metres left of the centreline. The env reports them as the
state's x and y, so the jump scoring, the reward (progress along s) and the experts work as on the straight courses.

Seeds: validation 10000-10019 (VAL_SEEDS), the gate's test loops 46000-46019 (TEST_SEEDS, gate only), practice from
100000. A loop seed names a different course than the same seed in course.make.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sdc.jumps import course as crs

BOX = 64.0            # m, side of the square heightfield (centred on the origin)
D = 0.05              # m, grid spacing (both ways)
N = int(round(BOX / D)) + 1
ZMAX = crs.ZMAX
DRIVE_W = crs.DRIVE_W
BANK_W, BANK_H = 0.4, 0.35      # the lane's banks: rise 0.35 m over 0.4 m (as on the straight courses)
STRIP = DRIVE_W / 2 + 1.0       # half-width of the strip of terrain shaped around the centreline
BASE = 0.45                     # m, ground level above the heightfield's base (room for banked turns)
LAP = (130.0, 175.0)            # m, lap length range
MIN_RADIUS = 3.0                # m, tightest corner on the centreline
STRAIGHT = 1 / 40.0             # |curvature| below this counts as straight
BANK_MAX = 12.0                 # deg, steepest banked turn
DS = 0.05                       # m, spacing of the stored centreline and height profile
START_X = crs.START_X           # the car starts this far past the line
N_JUMPS = 3
KINDS = crs.KINDS45
VAL_SEEDS = list(range(10000, 10020))
TEST_SEEDS = list(range(46000, 46020))   # gate only
TERRAIN = dict(nrow=N, ncol=N, rx=BOX / 2, ry=BOX / 2, cx=0.0, cy=0.0)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _resample(pts, ds):
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = int(round(s[-1] / ds))
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])], 1), float(s[-1])


def _blur(v, sigma_pts):
    """Circular Gaussian blur along axis 0."""
    h = int(4 * sigma_pts) + 1
    w = np.exp(-0.5 * (np.arange(-h, h + 1) / sigma_pts) ** 2)
    w /= w.sum()
    v = np.asarray(v, float)
    pad = np.concatenate([v[-h:], v, v[:h]])
    if v.ndim == 1:
        return np.convolve(pad, w, "valid")
    return np.stack([np.convolve(pad[:, j], w, "valid") for j in range(v.shape[1])], 1)


def _geometry(center):
    """Tangent heading and signed curvature (left turn > 0) of a closed centreline with even spacing."""
    n = len(center)
    d = np.roll(center, -1, 0) - np.roll(center, 1, 0)
    heading = np.arctan2(d[:, 1], d[:, 0])
    ds = np.linalg.norm(np.roll(center, -1, 0) - center, axis=1).mean()
    kappa = _wrap(np.roll(heading, -1) - np.roll(heading, 1)) / (2 * ds)
    return heading, kappa, ds * n


def _runs(mask):
    """Circular runs of True in mask: list of (start index, length)."""
    n = len(mask)
    if mask.all():
        return [(0, n)]
    out = []
    i0 = int(np.flatnonzero(~mask)[0])
    i = 0
    while i < n:
        j = (i0 + i) % n
        if mask[j]:
            k = 0
            while mask[(j + k) % n]:
                k += 1
            out.append((j, k))
            i += k
        else:
            i += 1
    return out


def _checked(center):
    """(center, curvature) recentred, or None if the outline breaks a rule (size, tightest corner, self-clearance)."""
    center = center - (center.max(0) + center.min(0)) / 2
    if np.abs(center).max() > BOX / 2 - STRIP - 0.5:
        return None
    _, kappa, _ = _geometry(center)
    kappa = _blur(kappa, 0.3 / DS)
    if np.abs(kappa).max() > 1 / MIN_RADIUS:
        return None
    # no two parts of the loop closer than two strips (points more than 12 m apart along it)
    c = center[::10]
    dist = np.linalg.norm(c[:, None] - c[None], axis=2)
    idx = np.arange(len(c))
    sep = np.abs(idx[:, None] - idx[None])
    sep = np.minimum(sep, len(c) - sep) * DS * 10
    if np.any(dist[sep > 12.0] < 2 * STRIP + 0.5):
        return None
    return center, kappa


def _centerline(rng):
    """A rounded random polygon (4-6 corners) scaled to a lap in LAP, with a chicane (a sideways S-bend, so the loop
    turns both ways) on its fourth-longest straight when there is one. None if it breaks a rule."""
    k = int(rng.integers(4, 7))
    ang = np.sort((np.arange(k) + rng.uniform(-0.3, 0.3, k)) * 2 * np.pi / k)
    rad = rng.uniform(0.5, 1.0, k)
    poly = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
    _, per = _resample(poly, 0.01)
    lap = rng.uniform(*LAP)
    sigma = rng.uniform(2.2, 3.4)
    dense, _ = _resample(poly * (lap / per) * 1.04, DS)         # the blur shortens the outline a little
    dense = _blur(dense, sigma / DS)
    if rng.random() < 0.5:
        dense = dense[::-1]
    center, length = _resample(dense, DS)
    got = _checked(center)
    if got is None:
        return None
    center, kappa = got
    runs = sorted(_runs(np.abs(kappa) < STRAIGHT), key=lambda r: -r[1])
    u_amp, side = rng.uniform(0.14, 0.2), rng.choice([-1.0, 1.0])
    if len(runs) > N_JUMPS and runs[N_JUMPS][1] * DS >= 13.0:
        a, m = runs[N_JUMPS]
        idx = (a + np.arange(m)) % len(center)
        heading, _, _ = _geometry(center)
        th = heading[idx[m // 2]]
        bump = side * u_amp * m * DS * np.sin(np.pi * np.arange(m) / m) ** 2
        center[idx] += bump[:, None] * np.array([-np.sin(th), np.cos(th)])
        center, length = _resample(center, DS)
        got = _checked(center)
        if got is None:
            return None
        center, kappa = got
    if not LAP[0] <= len(center) * DS <= LAP[1]:
        return None
    return center, kappa


@dataclass
class LoopCourse:
    seed: int
    center: np.ndarray       # (n, 2) centreline, DS apart, index 0 = the start/finish line
    heading: np.ndarray      # (n,) tangent heading, rad
    kappa: np.ndarray        # (n,) signed curvature, 1/m (left turn > 0)
    length: float            # lap length, m
    profile: np.ndarray      # (n,) centreline height above BASE: jumps
    rough: np.ndarray        # (n,) roughness and whoops along s
    bank: np.ndarray         # (n,) tan of the lane's cross slope (left side lower > 0)
    jumps: list = field(default_factory=list)
    whoops: tuple = (0.0, 0.0)   # (start, end) in s
    loop = True
    drive_w = DRIVE_W

    @property
    def finish(self):
        return self.length

    @property
    def ds(self):
        return self.length / len(self.center)

    def _at(self, arr, s):
        n = len(arr)
        q = (np.asarray(s, float) % self.length) / self.ds
        i = np.floor(q).astype(int) % n
        f = q - np.floor(q)
        return arr[i] * (1 - f) + arr[(i + 1) % n] * f

    def height(self, s):
        """Ground height on the centreline at s (s may run past one lap)."""
        return BASE + self._at(self.profile, s) + self._at(self.rough, s)

    def slope(self, s, h=0.1):
        return (self.height(np.asarray(s) + h) - self.height(np.asarray(s) - h)) / (2 * h)

    def curvature(self, s):
        return self._at(self.kappa, s)

    def pose(self, s, d=0.0):
        """World (X, Y, heading) at track coordinates (s, d)."""
        i = int(round((s % self.length) / self.ds)) % len(self.center)
        th = self.heading[i]
        x, y = self.center[i] + d * np.array([-np.sin(th), np.cos(th)])
        return float(x), float(y), float(th)

    def project(self, X, Y, hint=None, window=6.0):
        """Track coordinates of a world point: (s in [0, length), d, tangent heading, centreline index).
        `hint`: a centreline index near the answer (the car's last one); the search stays within `window` m of it,
        so a nearby part of the loop can't be mistaken for this one."""
        n = len(self.center)
        if hint is None:
            idx = np.arange(n)
        else:
            w = int(window / self.ds)
            idx = np.arange(hint - w, hint + w + 1) % n
        p = np.array([X, Y])
        i = int(idx[np.argmin(((self.center[idx] - p) ** 2).sum(1))])
        th = self.heading[i]
        t = np.array([np.cos(th), np.sin(th)])
        rel = p - self.center[i]
        along = float(rel @ t)
        d = float(rel @ np.array([-t[1], t[0]]))
        return (i * self.ds + along) % self.length, d, float(th), i

    def jump_at(self, s):
        """Index of the jump whose own stretch (1 m before its kicker to 1 m past its landing) contains s."""
        s = s % self.length
        for i, j in enumerate(self.jumps):
            if j.start - 1.0 <= s <= j.end + 1.0:
                return i
        return None

    def surface(self, i, d):
        """Height at centreline index i (array) and lateral offset d (array), broadcast together."""
        dc = np.clip(d, -DRIVE_W / 2, DRIVE_W / 2)
        bank = np.clip((np.abs(d) - DRIVE_W / 2) / BANK_W, 0, 1) * BANK_H
        return BASE + self.profile[i] + self.rough[i] + bank - dc * self.bank[i]

    def hfield(self) -> np.ndarray:
        """(N, N) elevations in [0, 1] (times ZMAX), rows along Y. Outside the lane's strip: a plateau at bank height."""
        if getattr(self, "_hf", None) is not None:
            return self._hf
        z = np.full((N, N), np.nan, np.float32)
        n = len(self.center)
        # splat the strip: fine samples along s and across d, each written to its nearest grid cell
        sub = 4
        q = np.arange(n * sub) / sub
        i0 = np.floor(q).astype(int) % n
        f = (q - np.floor(q))[:, None]
        c = self.center[i0] * (1 - f) + self.center[(i0 + 1) % n] * f
        th = self.heading[i0] + _wrap(self.heading[(i0 + 1) % n] - self.heading[i0]) * f[:, 0]
        nx, ny = -np.sin(th), np.cos(th)
        dd = np.arange(-STRIP, STRIP + 1e-9, 0.025)
        for d in dd:
            ix = np.rint((c[:, 0] + d * nx + BOX / 2) / D).astype(int)
            iy = np.rint((c[:, 1] + d * ny + BOX / 2) / D).astype(int)
            zz = (self.surface(i0, d) * (1 - f[:, 0]) + self.surface((i0 + 1) % n, d) * f[:, 0])
            z[iy, ix] = zz
        hole = np.isnan(z)
        for _ in range(2):   # the odd cell the splat missed inside the strip: the mean of its filled neighbours
            pad = np.pad(z, 1, constant_values=np.nan)
            nb = np.stack([pad[1 + a:N + 1 + a, 1 + b:N + 1 + b] for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b])
            cnt = (~np.isnan(nb)).sum(0)
            fill = hole & (cnt >= 5)
            z[fill] = np.nansum(nb[:, fill], 0) / cnt[fill]
            hole = np.isnan(z)
        z[hole] = BASE + BANK_H
        self._hf = (np.clip(z, 0, ZMAX) / ZMAX).astype(np.float32)
        return self._hf

    def to_dict(self):
        return dict(seed=self.seed, length=round(self.length, 2), jumps=[j.to_dict() for j in self.jumps],
                    whoops=[round(w, 2) for w in self.whoops])


def make(seed: int, kinds=KINDS, size=1.0) -> LoopCourse:
    """Generate the loop for `seed` (redrawn deterministically until every rule holds)."""
    rng = np.random.default_rng([int(seed), 46])
    for _ in range(5000):
        got = _centerline(rng)
        if got is None:
            continue
        center, kappa = got
        n = len(center)
        length = n * DS
        # straights: the longest takes the start line and jump 1 (it needs a run-up from a standing start)
        runs = sorted(_runs(np.abs(kappa) < STRAIGHT), key=lambda r: -r[1])
        if len(runs) < N_JUMPS or runs[0][1] * DS < 27.0 or runs[N_JUMPS - 1][1] * DS < 19.0:
            continue
        line = (runs[0][0] + int(1.0 / DS)) % n            # the line: 1 m into the longest straight
        center, kappa = np.roll(center, -line, 0), np.roll(kappa, -line)
        heading, _, _ = _geometry(center)
        off = int(1.0 / DS)
        straights = sorted([(0, runs[0][1] - off)] + [((a - line) % n, k) for a, k in runs[1:N_JUMPS]])   # driving order
        x = np.arange(n) * DS
        z = np.zeros(n)
        jumps, ok = [], True
        for m, (a, k) in enumerate(straights):
            s0, s1 = a * DS, (a + k) * DS
            run_up = 10.0 if m == 0 else 5.0               # straight before the kicker
            slack = (s1 - 4.0) - (s0 + run_up) - 12.0      # jumps are up to about 12 m long; 4 m straight after
            pos = s0 + run_up + rng.uniform(0, max(slack, 0.0))
            j = crs.build_jump(rng, x, z, pos, kinds, size)
            if j.end > s1 - 3.0 or j.end > length - 6.0:
                ok = False
                break
            jumps.append(j)
        if not ok:
            continue
        # banked turns: each corner gets its own bank (a third are flat), scaled by how tight the turn is
        bank = np.zeros(n)
        for a, k in _runs(np.abs(kappa) >= STRAIGHT):
            deg = 0.0 if rng.random() < 0.35 else rng.uniform(4.0, BANK_MAX)
            idx = (a + np.arange(k)) % n
            bank[idx] = np.tan(np.deg2rad(deg)) * np.clip(kappa[idx] * 8.0, -1, 1)
        bank = _blur(bank, 1.0 / DS)
        # roughness and one stretch of whoops, kept off the jumps (from 4 m before a kicker to 7 m past its landing)
        clear = np.ones(n, bool)
        for j in jumps:
            clear &= ~((x > j.start - 4.0) & (x < j.end + 7.0))
        clear &= (x > 4.0) & (x < length - 1.0)
        rough = np.zeros(n)
        for _ in range(6):
            wl = rng.uniform(1.5, 5.0)
            rough += rng.uniform(0.003, 0.008) * np.sin(2 * np.pi * x / wl + rng.uniform(0, 2 * np.pi))
        whoops = (0.0, 0.0)
        free = sorted(_runs(clear), key=lambda r: -r[1])
        if free and free[0][1] * DS > 9.0 and free[0][0] + free[0][1] <= n:
            a, k = free[0]
            wlen = rng.uniform(6.0, min(9.0, k * DS - 2.0))
            w0 = a * DS + 1.0 + rng.uniform(0, k * DS - 2.0 - wlen)
            amp, wl = rng.uniform(0.025, 0.045), rng.uniform(1.2, 1.8)
            m = (x >= w0) & (x <= w0 + wlen)
            env = np.sin(np.pi * (x[m] - w0) / wlen) ** 0.5
            rough[m] += amp * env * (1 - np.cos(2 * np.pi * (x[m] - w0) / wl))
            whoops = (float(w0), float(w0 + wlen))
        fade = _blur(clear.astype(float), 0.5 / DS)
        rough = rough * np.clip(fade * 2 - 1, 0, 1)
        return LoopCourse(seed, center, heading, kappa, length, z, rough, bank, jumps, whoops)
    raise RuntimeError(f"could not generate a loop for seed {seed}")


def cast(course, origin, vecs, cutoff, step=0.04):
    """Distances from `origin` along unit vectors `vecs` (n, 3) to the terrain, `cutoff` if nothing within it.
    Marches each ray over the height map in `step` m steps and interpolates the crossing. (MuJoCo's own ray cast
    takes about 3 ms per sensor read on a heightfield this large; this takes about 0.1 ms.)"""
    hf = course.hfield()
    t = np.arange(0.0, cutoff + step, step)
    p = origin[None, None, :] + vecs[:, None, :] * t[None, :, None]            # (n, T, 3)
    gx = np.clip((p[..., 0] + BOX / 2) / D, 0, N - 1.001)
    gy = np.clip((p[..., 1] + BOX / 2) / D, 0, N - 1.001)
    ix, iy = gx.astype(int), gy.astype(int)
    fx, fy = gx - ix, gy - iy
    h = ((hf[iy, ix] * (1 - fx) + hf[iy, ix + 1] * fx) * (1 - fy)
         + (hf[iy + 1, ix] * (1 - fx) + hf[iy + 1, ix + 1] * fx) * fy) * ZMAX
    gap = p[..., 2] - h                                                         # > 0: the ray is above the ground
    below = gap <= 0
    hit = below.any(1)
    k = np.maximum(below.argmax(1), 1)
    r = np.arange(len(vecs))
    g0, g1 = gap[r, k - 1], gap[r, k]
    frac = np.where(g0 > g1, g0 / np.maximum(g0 - g1, 1e-9), 0.0)
    dist = (k - 1 + np.clip(frac, 0, 1)) * step
    return np.where(hit, np.minimum(dist, cutoff), cutoff)
