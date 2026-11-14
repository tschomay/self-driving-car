"""E2: how well does the owner's complementary filter track pitch and roll from the realistic IMU?

  python -m sdc.phase45.e2 --out runs/p45-e2

The 4.5 expert drives the realistic car (all physics, noisy sensors) on the 20 validation courses (3 starts),
at nominal, training-range and gate-range physics. Every step logs the IMU reading the policy would see and the
true pitch and roll. Estimators, run on the same logs:
  gyro only          add up the raw gyro (no bias fix)
  gyro, bias fixed   the same, minus the still-car average
  accelerometer only the angle of the felt gravity, every step
  your filter        my_filter.AttitudeFilter (alpha 0.98, accelerometer ignored away from 1 g, bias fixed)
  ... always blend   your filter without choice 2 (the accelerometer is always blended in)
Errors in degrees (RMS) on the ground, in the air, at touchdown (pitch), and the worst error in a run.
Writes <out>/e2.json.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sdc.jumps import course as crs
from sdc.jumps import driver as dr
from sdc.jumps import env as je
from sdc.phase45 import my_filter as mf
from sdc.phase45.common import EXPERT

RANGES = ("nominal", "train", "gate")


def log_run(course, x0, d):
    b = je.JumpBatch(1, courses=[course], real="all", sensors="noisy", dr=d, fan=True, servo=True)
    c = b.cars[0]
    b._obs = c.reset(course, x0=x0)[None]
    ex = dr.ExpertDriver(**EXPERT)
    rows = []
    while True:
        obs, _, term, trunc, _ = b.step(ex(b))
        if term[0] or trunc[0]:
            break
        s = c.state()
        rows.append(np.concatenate([obs[0, :3] * 9.81, obs[0, 3:6] * 5.0, [s["pitch"], s["roll"], float(c.airborne)]]))
    return dict(rows=np.array(rows), still_acc=c.still_acc, still_gyro=c.still_gyro)


class GyroOnly:
    def __init__(self, fix_bias):
        self.fix = fix_bias

    def reset(self, still_gyro, still_acc):
        self.bias = still_gyro.mean(0) if self.fix else np.zeros(3)
        self.p, self.r = mf.accel_angles(still_acc.mean(0))

    def update(self, acc, gyro):
        g = gyro - self.bias
        self.p, self.r = self.p - g[1] * mf.DT, self.r + g[0] * mf.DT
        return self.p, self.r


class AccelOnly:
    def reset(self, still_gyro, still_acc):
        pass

    def update(self, acc, gyro):
        return mf.accel_angles(acc)


def estimators():
    return {"gyro only": GyroOnly(False), "gyro, bias fixed": GyroOnly(True), "accelerometer only": AccelOnly(),
            "your filter": mf.AttitudeFilter(), "... always blend": mf.AttitudeFilter(g_tol=np.inf)}


def score(run):
    R = run["rows"]
    air = R[:, 8] > 0.5
    touch = np.flatnonzero(air[:-1] & ~air[1:]) + 1
    out = {}
    for name, est in estimators().items():
        if isinstance(est, mf.AttitudeFilter):
            est.reset(run["still_gyro"], run["still_acc"])
        else:
            est.reset(run["still_gyro"], run["still_acc"])
        E = np.array([est.update(r[:3], r[3:6]) for r in R])
        err = np.degrees(E - R[:, 6:8])   # (T, 2): pitch, roll
        out[name] = dict(
            ground=np.sqrt((err[~air] ** 2).mean(0)).tolist(),
            air=np.sqrt((err[air] ** 2).mean(0)).tolist() if air.any() else [np.nan, np.nan],
            touchdown_pitch=np.abs(err[touch, 0]).tolist(),
            worst=np.abs(err).max(0).tolist(),
            end=np.abs(err[-1]).tolist())
    return out


def job(args):
    course, x0, d = args
    run = log_run(course, x0, d)
    return d, score(run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/p45-e2")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    jobs = [(c, x0, d) for d in RANGES for x0 in dr.STARTS for c in crs.VAL_SEEDS]
    res = {d: [] for d in RANGES}
    with Pool(a.procs) as p:
        for d, sc in p.imap_unordered(job, jobs):
            res[d].append(sc)
    summary = {}
    for d in RANGES:
        summary[d] = {}
        for name in estimators():
            rs = [r[name] for r in res[d]]
            summary[d][name] = dict(
                ground=np.sqrt(np.mean([np.square(r["ground"]) for r in rs], 0)).round(2).tolist(),
                air=np.sqrt(np.nanmean([np.square(r["air"]) for r in rs], 0)).round(2).tolist(),
                touchdown_pitch=round(float(np.mean([x for r in rs for x in r["touchdown_pitch"]])), 2),
                worst=np.median([r["worst"] for r in rs], 0).round(1).tolist(),
                end=np.median([r["end"] for r in rs], 0).round(2).tolist())
            print(d, f"{name:20s}", summary[d][name], flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, "e2.json").write_text(json.dumps(dict(summary=summary, runs=res)))


if __name__ == "__main__":
    main()
