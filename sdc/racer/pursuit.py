"""Hand-coded baseline driver: pure pursuit for steering, a braking-distance speed planner for throttle.

Unlike the RL policy, it cheats: it reads the track's centerline and curvature
directly instead of seeing only the 11 rays. That makes it a good yardstick for
lap time, which is what the Phase 1 gate compares against (RL ≤ 1.1× its lap time).

Steering (pure pursuit): pick a point on the centerline LOOKAHEAD metres ahead and
steer along the circular arc that passes through it.
Speed: for every centerline point in the next PLAN_DIST metres, the fastest speed
that corner allows is sqrt(margin·A_LAT / curvature); the car may go faster now only
if it can brake down to that speed in time: v² ≤ v_corner² + 2·BRAKE·distance.
"""

from __future__ import annotations

import numpy as np

from sdc.racer import sim
from sdc.racer import track as trk

PLAN_DIST = 60.0


class PurePursuit:
    def __init__(self, lat_margin: float = 0.95, brake_margin: float = 0.8, k_lookahead: float = 0.5,
                 min_lookahead: float = 4.0):
        self.lat_margin = lat_margin
        self.brake_margin = brake_margin
        self.k_lookahead = k_lookahead
        self.min_lookahead = min_lookahead

    def __call__(self, b: sim.RacerBatch) -> np.ndarray:
        N = trk.N_POINTS
        r = np.arange(b.n)
        # steering
        look = self.min_lookahead + self.k_lookahead * b.v
        tgt = (b.idx + np.ceil(look / b.ds).astype(int)) % N
        to = b.center[r, tgt] - b.pos
        alpha = np.arctan2(to[:, 1], to[:, 0]) - b.heading
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        dist = np.linalg.norm(to, axis=1)
        delta = np.arctan2(2 * sim.WHEELBASE * np.sin(alpha), dist)
        steer = np.clip(delta / sim.steer_limit(b.v), -1, 1)
        # speed
        n_ahead = int(np.ceil(PLAN_DIST / b.ds.min()))
        offs = np.arange(n_ahead)
        ahead = (b.idx[:, None] + offs[None]) % N
        kappa = np.abs(b.curv[r[:, None], ahead])
        v_corner = np.minimum(sim.V_MAX, np.sqrt(self.lat_margin * sim.A_LAT / np.maximum(kappa, 1e-6)))
        d = offs[None] * b.ds[:, None]
        v_allow = np.sqrt(v_corner ** 2 + 2 * self.brake_margin * sim.BRAKE * d)
        v_target = v_allow.min(axis=1)
        throttle = np.clip((v_target - b.v) * 1.0, -1, 1)
        return np.stack([steer, throttle], axis=1)
