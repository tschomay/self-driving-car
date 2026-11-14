"""Phase 4.5 E3c: the out-lap. Drive the course once at a steady speed, record each jump from the car's own sensors,
and turn that into a map for the timed lap (owner's choice, 2026-11-14: "C is the only one that's realistic").

Sensor-only. Everything here reads the parts-layout observation (what the policy sees) plus the still-car IMU
calibration, never the sim's true state; the true design speeds are used only as regression labels on training courses.

  Odometer      integrates the motor speed. In free fall (|accel| < FREE_G) the wheels spin up freely, so it holds
                the last ground speed instead.
  Jump record   a free-fall spell of at least MIN_AIR s: odometer at takeoff, ground speed at takeoff, the owner's
                filter pitch at takeoff (about the lip angle) and the steepest pitch on the kicker, airtime, the filter
                pitch at touchdown, and the peak accelerometer reading in the 0.1 s after touchdown.
  Map           per recorded jump: (odometer at takeoff, target speed). The target speed comes from a ridge regression
                of the record on the jump's design speed, fitted on training-course out-laps (like Phase 3's grip
                estimator: trained on the truth, used without it).

    python -m sdc.phase45.outlap fit --driver runs/p45-steady-bc-s0 --n 600     # fit the estimator
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sdc.jumps import course as crs
from sdc.jumps import env as je
from sdc.phase45 import my_filter as mf

G = 9.81
DT = 0.02
FREE_G = 0.4        # |accel| below this (in g) = free fall
MIN_AIR = 0.25      # s
I_ACC, I_GYRO, I_SPEED = slice(0, 3), slice(3, 6), 6   # parts-layout observation slots
FEATURES = ("v_takeoff", "pitch_takeoff", "pitch_max", "air", "pitch_touch", "acc_touch")
DEFAULT_SPEED = 7.4  # m/s for jumps the map doesn't know (the steady out-lap speed)
ESTIMATOR = Path("runs/p45-outlap-estimator.json")


class Odometer:
    """Distance along the course from the motor speed, holding the last ground speed while in free fall."""

    def __init__(self):
        self.odo, self.v_ground, self.free = 0.0, 0.0, False

    def update(self, obs):
        acc = np.asarray(obs[I_ACC]) * G
        v = float(obs[I_SPEED]) * 10.0
        self.free = np.linalg.norm(acc) / G < FREE_G
        if not self.free:
            self.v_ground = v
        self.odo += self.v_ground * DT
        return self.odo


class Recorder:
    """Feeds on one run's observations; records every jump."""

    def __init__(self, still_gyro, still_acc):
        self.odo = Odometer()
        self.filt = mf.AttitudeFilter(telemetry=True)
        self.filt.reset(still_gyro, still_acc)
        self.jumps, self._air, self._after, self._climb = [], None, None, []

    def update(self, obs):
        o = self.odo.update(obs)
        acc = np.asarray(obs[I_ACC]) * G
        pitch, _ = self.filt.update(acc, np.asarray(obs[I_GYRO]) * 5.0, speed=float(obs[I_SPEED]) * 10.0)
        self._climb = (self._climb + [pitch])[-40:]   # the last 0.8 s on the ground: the kicker
        if self.odo.free:
            if self._air is None:
                self._air = dict(odo=o, v_takeoff=self.odo.v_ground, pitch_takeoff=pitch,
                                 pitch_max=max(self._climb), n=0)
            self._air["n"] += 1
        elif self._air is not None:
            a = self._air
            self._air = None
            if a["n"] * DT >= MIN_AIR:
                a.update(air=a.pop("n") * DT, pitch_touch=pitch, acc_touch=np.linalg.norm(acc) / G)
                self._after = (a, 5)
                self.jumps.append(a)
        if self._after is not None and not self.odo.free:
            a, k = self._after
            a["acc_touch"] = max(a["acc_touch"], np.linalg.norm(acc) / G)
            self._after = (a, k - 1) if k > 1 else None


def drive(policy, course, x0=0.0, **env_kw):
    """One out-lap: returns (recorded jumps, the true course jumps the car took off from, for labels only)."""
    b = je.JumpBatch(1, courses=[course], **env_kw)
    c = b.cars[0]
    b._obs = c.reset(course, x0=x0)[None]
    if hasattr(policy, "reset"):
        policy.reset(1)
    rec = Recorder(c.still_gyro, c.still_acc)
    labels = []
    n_prev = 0
    while True:
        x_before = c.state()["x"]   # label bookkeeping only (true position at the step a jump is recorded)
        obs, _, term, trunc, _ = b.step(policy(b))
        if term[0] or trunc[0]:   # (the batch has already reset to a new run)
            break
        rec.update(obs[0])
        if rec._air is not None and rec._air.get("_x") is None:
            rec._air["_x"] = x_before
        if len(rec.jumps) > n_prev:
            n_prev = len(rec.jumps)
            k = c.course.jump_at(rec.jumps[-1].pop("_x", x_before))
            labels.append(None if k is None else c.course.jumps[k].design_speed)
    return rec.jumps, labels


def features(j):
    f = np.array([j[k] for k in FEATURES], float)
    return np.concatenate([[1.0], f, f[1:4] * f[0]])   # bias, raw features, a few speed interactions


def fit(rows, labels, ridge=1e-2):
    X = np.stack([features(j) for j in rows])
    y = np.array(labels)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    mu[0], sd[0] = 0.0, 1.0
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ y)
    return dict(w=w.tolist(), mu=mu.tolist(), sd=sd.tolist())


def predict(model, j):
    z = (features(j) - np.array(model["mu"])) / np.array(model["sd"])
    return float(np.clip(z @ np.array(model["w"]), crs.DESIGN_SPEED[0] - 0.5, crs.DESIGN_SPEED[1] + 0.5))


def make_map(jumps, model):
    return [(j["odo"], predict(model, j)) for j in jumps]


# ---- fitting the estimator ------------------------------------------------------------------------------
_POL = None


def _init(driver):
    global _POL
    import torch
    torch.set_num_threads(1)
    from sdc.phase4 import train as tr
    _POL, _ = tr.load(Path(driver))


def _job(args):
    course, x0, kw = args
    jumps, labels = drive(_POL, course, x0, **kw)
    return [(j, l) for j, l in zip(jumps, labels) if l is not None]


def collect(driver, courses, starts=(0.0,), dr="train", procs=4, phys_seed=0):
    kw = dict(real="all", sensors="parts", dr=dr, phys_seed=phys_seed)
    jobs = [(c, x0, kw) for c in courses for x0 in starts]
    with Pool(procs, initializer=_init, initargs=(driver,)) as p:
        return [r for rs in p.map(_job, jobs) for r in rs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit"])
    ap.add_argument("--driver", default="runs/p45-steady-bc-s0")
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    train_courses = list(range(100_000, 100_000 + a.n))   # practice courses (never validation or test)
    tr_rows = collect(a.driver, train_courses, dr="train")
    model = fit([j for j, _ in tr_rows], [l for _, l in tr_rows])
    out = dict(model=model, n_train=len(tr_rows))
    for name, dr in (("val_train", "train"), ("val_gate", "gate")):
        rows = collect(a.driver, crs.VAL_SEEDS, starts=(0.0, -0.3, 0.3), dr=dr)
        err = np.array([predict(model, j) - l for j, l in rows])
        base = np.array([np.mean([l for _, l in tr_rows]) - l for _, l in rows])
        out[name] = dict(n=len(rows), rmse=round(float(np.sqrt((err ** 2).mean())), 3),
                         rmse_constant=round(float(np.sqrt((base ** 2).mean())), 3))
        print(name, out[name], flush=True)
    ESTIMATOR.write_text(json.dumps(out, indent=1))
    print("train jumps", len(tr_rows), "->", ESTIMATOR)


if __name__ == "__main__":
    main()
