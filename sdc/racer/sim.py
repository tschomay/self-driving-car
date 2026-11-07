"""Batched 2D top-down racer: B cars, each on its own track, stepped together in numpy.

Physics: kinematic bicycle model with perfect grip (the car goes exactly where
its front wheels point, no sliding). One limit keeps speed meaningful: the
sideways acceleration in a turn, v²·curvature, can't exceed A_LAT, so at speed
the car can only steer gently. Fast through a tight corner is impossible; you
have to brake first.

Action (each in [-1, 1]):
  steer:    fraction of the steering available at the current speed (+ = left)
  throttle: + accelerates (up to ACCEL m/s²), − brakes (up to BRAKE m/s²)

Observation (12 numbers, each roughly in [0, 1]):
  11 ray distances, every 18° from −90° (right) to +90° (left), to the track
  edge, divided by RAY_RANGE; then speed / V_MAX.

Episode ends:
  terminated  car center more than OFFTRACK_LIMIT from the centerline (off the road)
  truncated   time limit (max_steps), or `laps` laps completed if set

Reward modes (`reward=`):
  progress  metres gained along the centerline this step / (V_MAX·DT): about 1 per
            step at top speed, negative when going backwards
  survival  +1 every step the car is still on the road
  center    +1 · (1 − |distance from centerline| / limit) every step on the road
Plus `crash_penalty` (subtracted once, on going off the road).
"""

from __future__ import annotations

import numpy as np

from sdc.racer import track as trk

DT = 0.1             # seconds per step (10 Hz control)
WHEELBASE = 2.5      # metres
STEER_MAX = np.radians(30)
V_MAX = 25.0         # m/s (90 km/h)
ACCEL = 6.0          # m/s²
BRAKE = 12.0         # m/s²
A_LAT = 8.0          # m/s², about 0.8 g of cornering
CAR_LEN, CAR_WID = 4.0, 1.8
OFFTRACK_LIMIT = trk.WIDTH / 2 - CAR_WID / 2   # 3.1 m from the centerline
N_RAYS = 11
RAY_ANGLES = np.radians(np.linspace(-90, 90, N_RAYS))
RAY_RANGE = 40.0
RAY_WINDOW = 110     # centerline points searched either side of the car for ray hits
OBS_DIM = N_RAYS + 1
REWARDS = ("progress", "survival", "center")


def steer_limit(v):
    """Largest steering angle allowed at speed v (perfect grip, but sideways accel ≤ A_LAT)."""
    v2 = np.maximum(np.asarray(v, dtype=float) ** 2, 1e-6)
    return np.minimum(STEER_MAX, np.arctan(WHEELBASE * A_LAT / v2))


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class RacerBatch:
    """B independent cars. Call reset() once; step() auto-resets finished cars.

    tracks: "random" (a fresh training track, seed < 10_000, on every reset), or a
    list of seeds to cycle through (car i starts on tracks[i % len]).
    start: "random" (random point on the track) or "line" (start line, index 0).
    """

    def __init__(self, n: int, tracks="random", reward: str = "progress", crash_penalty: float = 0.0,
                 max_steps: int = 1000, laps: int | None = None, start: str = "random", seed: int = 0):
        assert reward in REWARDS, reward
        self.n = n
        self.track_spec = tracks
        self.reward_mode = reward
        self.crash_penalty = crash_penalty
        self.max_steps = max_steps
        self.laps = laps
        self.start = start
        self.rng = np.random.default_rng(seed)
        self._cache: dict[int, trk.Track] = {}
        self._resets = 0
        N = trk.N_POINTS
        self.center = np.zeros((n, N, 2))
        self.tangent = np.zeros((n, N, 2))
        self.normal = np.zeros((n, N, 2))
        self.curv = np.zeros((n, N))
        self.left = np.zeros((n, N, 2))
        self.right = np.zeros((n, N, 2))
        self.length = np.ones(n)
        self.ds = np.ones(n)
        self.seeds = np.zeros(n, dtype=int)
        self.pos = np.zeros((n, 2))
        self.heading = np.zeros(n)
        self.v = np.zeros(n)
        self.idx = np.zeros(n, dtype=int)
        self.lateral = np.zeros(n)
        self.progress = np.zeros(n)   # metres along the centerline since the episode started
        self.steps = np.zeros(n, dtype=int)
        self.ep_return = np.zeros(n)
        self.steer = np.zeros(n)      # last applied steering angle, radians
        self.rays = np.zeros((n, N_RAYS))
        self._along = np.zeros(n)     # position along the nearest centerline segment

    # ---- tracks -------------------------------------------------------------------------
    def track(self, seed: int) -> trk.Track:
        if seed not in self._cache:
            if len(self._cache) > 2000:
                self._cache.clear()
            self._cache[seed] = trk.generate(seed)
        return self._cache[seed]

    def _pick_seed(self, i: int) -> int:
        if self.track_spec == "random":
            return int(self.rng.integers(0, trk.HELD_OUT_BASE))
        seeds = list(self.track_spec)
        if self._resets < self.n:   # first reset: car i gets track i
            return seeds[i % len(seeds)]
        return seeds[int(self.rng.integers(len(seeds)))]

    def _reset_car(self, i: int):
        t = self.track(self._pick_seed(i))
        self._resets += 1
        self.seeds[i] = t.seed
        self.center[i], self.tangent[i], self.normal[i] = t.center, t.tangent, t.normal
        self.curv[i], self.left[i], self.right[i] = t.curvature, t.left, t.right
        self.length[i], self.ds[i] = t.length, t.ds
        k = 0 if self.start == "line" else int(self.rng.integers(trk.N_POINTS))
        self.idx[i] = k
        self.pos[i] = t.center[k]
        self.heading[i] = np.arctan2(t.tangent[k, 1], t.tangent[k, 0])
        self.v[i] = 0.0
        self.lateral[i] = 0.0
        self.progress[i] = 0.0
        self.steps[i] = 0
        self.ep_return[i] = 0.0
        self.steer[i] = 0.0
        self._along[i] = 0.0

    def reset(self) -> np.ndarray:
        self._resets = 0
        for i in range(self.n):
            self._reset_car(i)
        self._cast_rays()
        return self.obs()

    # ---- geometry -----------------------------------------------------------------------
    def _locate(self):
        """Update nearest centerline index, lateral offset; return arc-length change (m)."""
        N = trk.N_POINTS
        offs = np.arange(-6, 16)
        cand = (self.idx[:, None] + offs[None]) % N
        rows = np.arange(self.n)[:, None]
        d = np.linalg.norm(self.center[rows, cand] - self.pos[:, None], axis=2)
        new = cand[np.arange(self.n), np.argmin(d, axis=1)]
        r = np.arange(self.n)
        rel = self.pos - self.center[r, new]
        along_new = np.einsum("ij,ij->i", rel, self.tangent[r, new])
        self.lateral = np.einsum("ij,ij->i", rel, self.normal[r, new])
        step_idx = (new - self.idx + N // 2) % N - N // 2
        ds = step_idx * self.ds + along_new - self._along
        self.idx = new
        self._along = along_new
        return ds

    def _cast_rays(self):
        N = trk.N_POINTS
        offs = np.arange(-RAY_WINDOW, RAY_WINDOW)
        seg_i = (self.idx[:, None] + offs[None]) % N
        rows = np.arange(self.n)[:, None]
        a = np.concatenate([self.left[rows, seg_i], self.right[rows, seg_i]], axis=1)          # (B, M, 2)
        b = np.concatenate([self.left[rows, (seg_i + 1) % N], self.right[rows, (seg_i + 1) % N]], axis=1)
        ang = self.heading[:, None] + RAY_ANGLES[None]
        d = np.stack([np.cos(ang), np.sin(ang)], axis=2)                                      # (B, K, 2)
        e = (b - a)[:, None]                                                                   # (B, 1, M, 2)
        ap = (a - self.pos[:, None])[:, None]                                                  # (B, 1, M, 2)
        dd = d[:, :, None]                                                                     # (B, K, 1, 2)
        denom = _cross(dd, e)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = _cross(ap, e) / denom
            u = _cross(ap, dd) / denom
        hit = (np.abs(denom) > 1e-9) & (t >= 0) & (u >= 0) & (u <= 1)
        t = np.where(hit, t, np.inf)
        self.rays = np.minimum(t.min(axis=2), RAY_RANGE)

    def obs(self) -> np.ndarray:
        return np.concatenate([self.rays / RAY_RANGE, (self.v / V_MAX)[:, None]], axis=1).astype(np.float32)

    # ---- dynamics -----------------------------------------------------------------------
    def step(self, action: np.ndarray):
        """Advance every car one step. Returns obs, reward, terminated, truncated, infos.

        Cars that finished are reset before returning; their final observation is in
        infos[i]["terminal_observation"] and episode stats in infos[i]["episode"].
        """
        a = np.clip(np.asarray(action, dtype=float).reshape(self.n, 2), -1, 1)
        self.steer = a[:, 0] * steer_limit(self.v)
        acc = np.where(a[:, 1] >= 0, a[:, 1] * ACCEL, a[:, 1] * BRAKE)
        v_old = self.v
        self.v = np.clip(self.v + acc * DT, 0, V_MAX)
        v_mid = 0.5 * (v_old + self.v)
        self.heading = self.heading + v_mid / WHEELBASE * np.tan(self.steer) * DT
        self.pos = self.pos + v_mid[:, None] * np.stack([np.cos(self.heading), np.sin(self.heading)], 1) * DT
        ds = self._locate()
        self.progress += ds
        self.steps += 1
        off = np.abs(self.lateral) > OFFTRACK_LIMIT

        if self.reward_mode == "progress":
            r = ds / (V_MAX * DT)
        elif self.reward_mode == "survival":
            r = np.ones(self.n)
        else:
            r = 1 - np.minimum(np.abs(self.lateral) / OFFTRACK_LIMIT, 1)
        r = np.where(off, -self.crash_penalty, r)
        self.ep_return += r
        lap_done = (self.progress >= self.length * self.laps) if self.laps else np.zeros(self.n, bool)
        terminated = off
        truncated = ~off & ((self.steps >= self.max_steps) | lap_done)

        self._cast_rays()
        obs = self.obs()
        infos = [{} for _ in range(self.n)]
        done = terminated | truncated
        for i in np.flatnonzero(done):
            infos[i] = dict(
                terminal_observation=obs[i].copy(),
                episode=dict(r=float(self.ep_return[i]), l=int(self.steps[i])),
                laps=float(self.progress[i] / self.length[i]),
                crashed=bool(terminated[i]),
                track=int(self.seeds[i]),
                mean_speed=float(self.progress[i] / (self.steps[i] * DT)),
                final_state=dict(pos=self.pos[i].copy(), heading=float(self.heading[i]), v=float(self.v[i]),
                                 rays=self.rays[i].copy(), lateral=float(self.lateral[i])),
                **{"TimeLimit.truncated": bool(truncated[i])},
            )
            self._reset_car(i)
        if done.any():
            self._cast_rays()
            obs[done] = self.obs()[done]
        return obs, r.astype(np.float32), terminated, truncated, infos
