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


# ---- the walk-through out-lap (owner, 2026-11-14: "just roll at a slow pace to learn the track") ----------------
I_SIDE = slice(72, 74)   # parts layout: VL53L1X left, right (/ 3 m)
CRAWL_V = 2.0


class Crawler:
    """Hand-coded, sensor-only crawl: hold CRAWL_V on the motor speed, centre on the two side sensors.
    (A careful driver walking the track; it never reads the sim's state.)"""

    def __init__(self, v=CRAWL_V, k_steer=1.5, k_head=6.0):
        self.v, self.k, self.kd = v, k_steer, k_head
        self.prev = None
        self.i = 0.0

    def reset(self, n):
        self.prev, self.i = None, 0.0

    def act(self, obs):
        left, right = (np.asarray(obs[I_SIDE]) * 3.0).tolist()
        off = 0.5 * (left - right)                     # > 0: car is right of centre
        d_off = 0.0 if self.prev is None else (off - self.prev) / DT
        self.prev = off
        steer = np.clip(self.k * off + self.kd * d_off * DT * 5, -0.5, 0.5)
        e = self.v - float(obs[I_SPEED]) * 10.0
        self.i = float(np.clip(self.i + e * DT, -2, 3))
        throttle = np.clip(0.6 * e + 0.15 + 0.3 * self.i, -0.6, 1.0)
        return np.array([steer, throttle], np.float32)

    def __call__(self, b):
        return np.stack([self.act(o) for o in b.obs()])


def walk(course, x0=0.0, max_steps=4000, filter_kw=None, **env_kw):
    """Crawl the course once; return the height profile measured from the car's own sensors:
    odometer (m) and height (m) per step, plus the run's ending (labels come from the caller, not from here)."""
    b = je.JumpBatch(1, courses=[course], max_steps=max_steps, kinds=crs.KINDS45, **env_kw)
    c = b.cars[0]
    b._obs = c.reset(course, x0=x0)[None]
    drv = Crawler()
    odo, filt = Odometer(), mf.AttitudeFilter(**(filter_kw or dict(telemetry=True, window=25)))
    filt.reset(c.still_gyro, c.still_acc)
    xs, zs, z, xh = [0.0], [0.0], 0.0, 0.0
    true_x = [c.state()["x"]]   # labels only (never used to build the profile)
    reason = None
    while True:
        obs, _, term, trunc, infos = b.step(drv(b))
        if term[0] or trunc[0]:
            reason = infos[0]["episode"]["reason"]
            break
        o = obs[0]
        before = odo.odo
        odo.update(o)
        pitch, _ = filt.update(np.asarray(o[I_ACC]) * G, np.asarray(o[I_GYRO]) * 5.0, speed=float(o[I_SPEED]) * 10.0)
        ds = odo.odo - before
        z += np.sin(pitch) * ds
        xh += np.cos(pitch) * ds        # horizontal distance (the odometer measures along the slopes)
        xs.append(xh)
        zs.append(z)
        true_x.append(c.state()["x"])
    return np.array(xs), np.array(zs), reason, np.array(true_x)


def find_jumps(xs, zs, min_h=0.12):
    """Jumps in a measured profile: each is a stretch standing at least min_h above the ground before it.
    Per jump: lip (end of the first climb), its height and steepest climb, where the final descent begins (the
    landing), its height and length. All in the out-lap's own odometer metres."""
    grid = np.arange(0, xs[-1], 0.05)
    z = np.interp(grid, xs, zs)
    z = np.convolve(np.pad(z, 3, mode="edge"), np.ones(7) / 7, mode="valid")   # 0.35 m smoothing
    slope = np.gradient(z, grid)
    # ground level: the lowest point within ±4 m (wider than any jump), smoothed; it follows the slow height drift
    w = 80
    pad = np.pad(z, w, mode="edge")
    ground = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1).min(1)
    ground = np.convolve(np.pad(ground, 20, mode="edge"), np.ones(41) / 41, mode="valid")
    above = z - ground
    out, i = [], 0
    while i < len(grid):
        if above[i] < min_h:
            i += 1
            continue
        s0 = i
        while s0 > 0 and above[s0 - 1] > 0.03:             # back to where the climb began
            s0 -= 1
        e = i
        while e < len(grid) - 1 and above[e] > 0.03:      # forward to back on the ground
            e += 1
        base = 0.0
        zz, ss = above[s0:e + 1], slope[s0:e + 1]
        top = int(np.argmax(ss))                           # steepest point of the first climb
        flat = np.flatnonzero(ss[top:] < 0.25 * ss[top])
        lip = top + (int(flat[0]) if len(flat) else int(np.argmax(zz[top:])))
        rising = np.flatnonzero(ss > -0.1)                 # the final descent starts after the last non-descending point
        land = max(int(rising[-1]) if len(rising) else lip, lip)
        out.append(dict(start=float(grid[s0]), lip=float(grid[s0 + lip]), h=float(zz[lip]),
                        angle=float(np.degrees(np.arctan(ss[top]))), land=float(grid[s0 + land]),
                        lh=float(zz[land]), ll=float(grid[e] - grid[s0 + land]), end=float(grid[e])))
        i = e + 1
    return out


def jump_features(j):
    d = j["land"] - j["lip"]
    return np.array([1.0, j["h"], j["angle"], d, j["lh"], j["ll"], d * j["h"]])
