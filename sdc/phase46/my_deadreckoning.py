"""The owner's dead-reckoning estimator (learner-owned, Phase 4.6): where is the car on the loop, from its own sensors?

Inputs each control step (50 Hz), all from the parts-list sensors:
  gyro (rad/s, body frame; z = turning left), motor speed (m/s), and the owner's attitude filter's pitch and roll.
Output: position (x, y) in metres and heading in radians, in a frame that starts at the start line: x along the
track at the line, y to the left. It starts at (0, 0, 0) and the finish-line sensor says when a lap is complete.

The map of the turns comes from this: the heading's change per metre travelled is the track's curvature.

Owner's choices (2026-11-16, "Agree" to the recommended set): gyro heading, tilt-corrected, distance held in
free fall, closing error spread along the lap at the finish line. Details the agent filled in are marked "agent".
"""

from __future__ import annotations

import numpy as np

DT = 0.02

# Owner's choice 1 - heading: where does the turn rate come from?
#   "gyro"      integrate the gyro's z rate (minus its bias measured while still). Drifts slowly with the bias.
#   "steering"  from the steering command and speed (bicycle model). No drift, but wrong when the tires slide.
#   "blend"     gyro for the short term, pulled toward the steering model over a few seconds.
HEADING = "gyro"

# Owner's choice 2 - tilt: on a banked turn or a ramp the gyro's z axis leans with the car, so it reads less than the
# true turn rate. True = use the attitude filter's pitch and roll to project the gyro onto the vertical;
# False = use the z reading as it is.
TILT_CORRECT = True

# Owner's choice 3 - distance: motor speed x time overcounts when the wheels spin in the air or slip.
#   "raw"    motor speed x time, always.
#   "hold"   in free fall, keep the last speed measured on the ground (Phase 4.5's odometer).
DISTANCE = "hold"

# Owner's choice 4 - the finish line: after a lap the estimate should be back at (0, 0) facing 0, and it won't be.
#   "reset"   snap back to (0, 0, 0) at the line; the lap's map keeps its error.
#   "spread"  also go back over the lap just driven and spread the closing error evenly along it, so the mapped
#             loop closes.
AT_LINE = "spread"


def ready():
    return None not in (HEADING, TILT_CORRECT, DISTANCE, AT_LINE)


G = 9.81
FREE_G = 0.4          # |accel| below this (in g) = free fall (agent: Phase 4.5's odometer threshold)
WHEELBASE = 0.30      # m (for the "steering" heading model)
STEER_MAX = 0.4       # rad at full lock


class DeadReckoning:
    def __init__(self, heading=None, tilt_correct=None, distance=None, at_line=None, start_x=1.5, dt=DT):
        """Defaults are the owner's choices; the arguments exist so E2 can compare the alternatives."""
        self.heading_src = HEADING if heading is None else heading
        self.tilt = TILT_CORRECT if tilt_correct is None else tilt_correct
        self.distance = DISTANCE if distance is None else distance
        self.at_line_mode = AT_LINE if at_line is None else at_line
        self.start_x, self.dt = start_x, dt
        self.reset()

    def reset(self, still_gyro=None):
        """Start of a run: the car stands start_x metres past the line, facing along the track.
        still_gyro: (n, 3) gyro readings while standing still; their mean is the bias (as in the attitude filter)."""
        self.bias = np.zeros(3) if still_gyro is None else np.mean(still_gyro, axis=0)
        self.x, self.y, self.h, self.s = self.start_x, 0.0, 0.0, self.start_x
        self.v_ground, self._cos = 0.0, 1.0
        self.path = [(self.s, self.x, self.y, self.h)]

    def update(self, gyro, speed, pitch=0.0, roll=0.0, acc=None, steer=0.0):
        """One control step. gyro (rad/s, body), motor speed (m/s), the attitude filter's pitch and roll (rad),
        acc (m/s², to spot free fall), steer (last steering command, -1..1). Returns (x, y, heading)."""
        g = np.asarray(gyro, float) - self.bias
        # owner, choice 2: project the body's turn rate onto the vertical (the standard Euler-angle formula)
        gyro_rate = (g[1] * np.sin(roll) + g[2] * np.cos(roll)) / np.cos(pitch) if self.tilt else g[2]
        # owner, choice 3: in free fall the wheels spin freely, so keep the last speed measured on the ground
        free = acc is not None and np.linalg.norm(acc) / G < FREE_G
        if self.distance == "raw" or not free:
            self.v_ground = float(speed)
        # agent: only the horizontal part of the distance counts. On the ground that is cos(pitch). In the air
        # (2026-11-17, after E2) the car keeps the horizontal speed it left the lip with, so the pitch at takeoff is
        # used for the whole flight: with the current pitch the held speed overcounted by about 15%.
        if not free:
            self._cos = float(np.cos(pitch))
        ds = self.v_ground * self.dt * self._cos
        # owner, choice 1: the gyro carries the heading
        model_rate = self.v_ground * np.tan(steer * STEER_MAX) / WHEELBASE
        if self.heading_src == "gyro":
            rate = gyro_rate
        elif self.heading_src == "steering":
            rate = 0.0 if free else model_rate
        else:   # blend: the gyro's rate, with the heading pulled toward the steering model's over about 3 s
            self._hm = getattr(self, "_hm", 0.0) + (0.0 if free else model_rate) * self.dt
            rate = gyro_rate + (self._hm - self.h) / 3.0
        self.h += float(rate) * self.dt
        self.x += ds * np.cos(self.h)
        self.y += ds * np.sin(self.h)
        self.s += ds
        self.path.append((self.s, self.x, self.y, self.h))
        return self.x, self.y, self.h

    def at_line(self):
        """The finish-line sensor fired: one lap done. Returns (path, closing error) where path is an (n, 4) array
        of (distance, x, y, heading) for the lap just driven and the closing error is (metres, radians) before any
        correction. Owner, choice 4 ("spread"): the heading error is spread evenly along the lap's distance, the
        positions are re-integrated from the corrected headings, and what position error remains is spread evenly
        too, so the mapped loop closes at the line. The live estimate then restarts from the line."""
        p = np.array(self.path)
        turns = np.round(p[-1, 3] / (2 * np.pi))
        err_h = p[-1, 3] - 2 * np.pi * turns          # the lap must turn a whole number of times
        err_xy = np.hypot(p[-1, 1], p[-1, 2])
        if self.at_line_mode == "spread":
            f = (p[:, 0] - p[0, 0]) / max(p[-1, 0] - p[0, 0], 1e-9)
            h = p[:, 3] - err_h * f
            ds = np.diff(p[:, 0])
            x = p[0, 1] + np.concatenate([[0], np.cumsum(ds * np.cos(h[1:]))])
            y = p[0, 2] + np.concatenate([[0], np.cumsum(ds * np.sin(h[1:]))])
            x, y = x - x[-1] * f, y - y[-1] * f
            p = np.stack([p[:, 0], x, y, h], 1)
        self.x, self.y, self.h, self.s = 0.0, 0.0, 0.0, 0.0
        self.path = [(0.0, 0.0, 0.0, 0.0)]
        return p, (float(err_xy), float(err_h))


def curvature(path, smooth=1.0, ds=0.25):
    """The map of the turns from a lap's path: (distance grid, curvature 1/m) = heading change per metre,
    smoothed over `smooth` metres (agent)."""
    s, h = path[:, 0], path[:, 3]
    grid = np.arange(s[0], s[-1], ds)
    hg = np.interp(grid, s, h)
    k = np.gradient(hg, grid)
    n = max(int(smooth / ds), 1)
    return grid, np.convolve(np.pad(k, n, mode="edge"), np.ones(2 * n + 1) / (2 * n + 1), mode="valid")
