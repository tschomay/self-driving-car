"""Phase 4.6: drive one lap and keep what the car sensed, with the truth alongside (for scoring and labels only).

record() returns per-step arrays: obs (the parts-layout observation, what the policy sees), and truth: track
position s, offset d, and the pose in the line's frame (x along the track at the line, y to the left, heading).
"""

from __future__ import annotations

import numpy as np

from sdc.jumps import env as je
from sdc.phase46 import common as cm

# parts-layout observation slots (sensors="parts", attitude=True)
I_ACC, I_GYRO, I_SPEED, I_TF = slice(0, 3), slice(3, 6), 6, 7
I_TOF, I_SIDE, I_ACT, I_ATT = slice(8, 72), slice(72, 74), slice(74, 76), slice(76, 78)


SIDE_H = 0.10        # m, the side sensors' height above the ground
HALF_LANE = 1.6      # m, from the lane's middle to where a level side beam meets the bank


class LoopCrawler:
    """A sensor-only crawl driver for the out-lap on loops (a careful driver walking the track; it never reads the
    sim's state). Phase 4.5's crawler steered on the change of the side ranges from step to step, which is mostly
    noise; on loops that sawing stalled it on kickers and threw it off in turns. This one:
      steering  toward the middle between the two side ranges (smoothed), damped by how fast that offset is
                changing over the last 0.2 s (the car's heading relative to the lane) and a little by the gyro. A side
                range that is missing, or that is reading the ground because the body is rolled (judged from the
                attitude filter's roll), is ignored and the other side is used alone. On a ramp it holds straight.
      speed     holds V on the motor speed, adds throttle for the slope (from the attitude filter's pitch), and
                speeds up to V_RAMP while climbing, to carry it over the lip."""

    V, V_RAMP, RAMP = 2.0, 3.0, np.deg2rad(10)

    def __init__(self, k_off=0.8, k_rate=0.6, k_gyro=0.05, v=None, lock=0.4):
        self.k_off, self.k_rate, self.k_gyro, self.lock = k_off, k_rate, k_gyro, lock
        self.v = self.V if v is None else v
        self.reset(1)

    def reset(self, n):
        self.off, self.i, self.boost, self.hist, self.steer = 0.0, 0.0, 0, [], 0.0

    def act(self, obs):
        left, right = (np.asarray(obs[I_SIDE]) * 3.0).tolist()
        pitch = float(obs[I_ATT][0]) * 0.5
        gyro_z = float(obs[I_GYRO][2]) * 5.0
        speed = float(obs[I_SPEED]) * 10.0
        roll = float(obs[I_ATT][1]) * 0.5
        # a side range is usable if it found something, and that something isn't the ground: with the body rolled,
        # the low side's beam hits the ground at about (sensor height / sin roll) before it reaches the bank
        reach = SIDE_H / max(abs(np.sin(roll)), 1e-3)
        ok_l = left < 2.9 and not (roll < 0 and left > 0.8 * reach)     # roll < 0: the left side is low
        ok_r = right < 2.9 and not (roll > 0 and right > 0.8 * reach)
        if abs(pitch) >= self.RAMP or not (ok_l or ok_r):
            self.off *= 0.9                                              # on a ramp, or blind: hold straight
        else:                                                            # > 0: the car is right of the middle
            off = 0.5 * (left - right) if ok_l and ok_r else (left - HALF_LANE if ok_l else HALF_LANE - right)
            self.off += 0.3 * (off - self.off)
        self.hist = (self.hist + [self.off])[-11:]
        rate = (self.hist[-1] - self.hist[0]) / 0.2 if len(self.hist) == 11 else 0.0    # m/s, over the last 0.2 s
        # gentle hands: hard steering rolls the body, and a rolled body's side beams read the ground (E1 of 4.5)
        want = np.clip(self.k_off * self.off + self.k_rate * rate - self.k_gyro * gyro_z, -self.lock, self.lock)
        self.steer += float(np.clip(want - self.steer, -0.04, 0.04))
        steer = self.steer
        if pitch > np.deg2rad(6):                                        # climbing: carry it over the lip
            self.boost = 25
        self.boost = max(self.boost - 1, 0)
        e = (self.V_RAMP if self.boost else self.v) - speed
        self.i = float(np.clip(self.i + e * 0.02, -2, 3))
        throttle = np.clip(0.6 * e + 0.15 + 0.3 * self.i + 1.2 * np.sin(pitch), -0.6, 1.0)
        return np.array([steer, throttle], np.float32)

    def __call__(self, b):
        return np.stack([self.act(o) for o in b.obs()])


def record(policy, course, x0=0.0, dr="gate", phys_seed=0, max_steps=None, **kw):
    env = {**cm.CAR, "dr": dr, "phys_seed": phys_seed, **kw}
    if max_steps:
        env["max_steps"] = max_steps
    b = je.JumpBatch(1, courses=[course], **env)
    c = b.cars[0]
    b._obs = c.reset(course, x0=x0)[None]
    if hasattr(policy, "reset"):
        policy.reset(1)
    X0, Y0, th0 = c.course.pose(0.0)
    # taken now: when the lap ends the batch resets the car for another run, with a new standstill calibration
    still_gyro, still_acc, length = c.still_gyro.copy(), c.still_acc.copy(), c.course.length
    obs, truth = [b._obs[0].copy()], []

    def true_pose():
        st = c.state()
        dx, dy = st["X"] - X0, st["Y"] - Y0
        return (st["x"], st["y"], np.cos(th0) * dx + np.sin(th0) * dy, -np.sin(th0) * dx + np.cos(th0) * dy,
                st["heading"] - th0, st["v_fwd"])

    truth.append(true_pose())
    while True:
        o, _, term, trunc, infos = b.step(policy(b))
        if term[0] or trunc[0]:
            ep = infos[0]["episode"]
            break
        obs.append(o[0].copy())
        truth.append(true_pose())
    return dict(course=course, x0=x0, obs=np.array(obs), truth=np.array(truth), still_gyro=still_gyro,
                still_acc=still_acc, reason=ep["reason"], episode=ep, length=length)
