"""Phase 3.5 E6: a brake test at the start of the out-lap anchors the grip estimate.

    python -m sdc.phase35.braketest calibrate                  # fit grip ≈ K × locked deceleration (training range)
    python -m sdc.phase35.braketest eval --run runs/p35-e5-lo-s0

The brake test: the car's own steering, throttle overridden. Accelerate until the speed reading passes V_TEST,
then full brake for up to BRAKE_STEPS steps: the wheels lock and the accelerometer reads the sliding
deceleration, which is proportional to grip (a sliding tire gives SLIDE_FRAC of its peak μ·g). Readings taken
after SETTLE steps (control delay and lock) while the speed reading is above V_MIN are averaged. K comes from a
regression on training-range physics only (an engineering calibration, as one would do on a test track).

After the test, the grip level comes from the test (g_bt) and the 1 s window estimator only says where the grip
changes: its reading relative to the lap's median so far, clipped to [DIRT_FLOOR, 1]. Grip input =
g_bt × that ratio; the map records the same per 5 m bin on the out-lap and is read 5–40 m ahead on the timed lap
(as in E5). Before the test finishes, the grip input is the estimate capped at 0.7 (cautious).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sdc.phase3 import robust
from sdc.phase35 import selfmap as sm
from sdc.racer import dynamics as dy
from sdc.racer import sim

V_TEST = 8.0       # m/s, start braking above this speed reading
BRAKE_STEPS = 10   # at most 1 s of full brake
SETTLE = 3         # steps before readings count (delay up to 0.12 s, then the wheels lock)
V_MIN = 1.5        # m/s, ignore readings once nearly stopped
DIRT_FLOOR = 0.6
CAL = Path("runs/p35-e6-calibration.json")
ACC_X = dy.OBS_DIM   # accelerometer forward reading / G (first extra sensor)


class BrakeTestDriver(sm.SelfMapDriver):
    def __init__(self, net, base, est, k=None):
        super().__init__(net, base, est)
        self.k = k if k is not None else json.loads(CAL.read_text())["k"]

    def reset_state(self, n):
        super().reset_state(n)
        self.phase = np.zeros(n, int)        # 0 speed up, 1 braking, 2 done
        self.brake_t = np.zeros(n, int)
        self.reads = [[] for _ in range(n)]
        self.g_bt = np.full(n, np.nan)
        self.ests = [[] for _ in range(n)]

    def ratio(self, i, g):
        ref = np.median(self.ests[i]) if self.ests[i] else g
        return float(np.clip(g / max(ref, 1e-3), DIRT_FLOOR, 1.0))

    def __call__(self, b):
        if self.n != b.n or (b.steps == 0).all():
            self.reset_state(b.n)
        o = b.obs()
        lap_now = np.floor(b.progress / b.length + 1e-9).astype(int)
        for i in np.flatnonzero(lap_now > self.lap):
            if self.lap[i] == 0:
                self.out_len[i] = self.odo[i]
            self.odo[i] = 0.0
        self.lap = np.maximum(self.lap, lap_now)
        est = self.grip_estimate(o, b)
        x = o[:, :self.base].copy()
        speed = o[:, sim.N_RAYS] * sim.V_MAX
        override = np.full(b.n, np.nan)
        for i in range(b.n):
            if self.phase[i] == 0:
                override[i] = 1.0 if speed[i] < V_TEST else -1.0
                if speed[i] >= V_TEST:
                    self.phase[i] = 1
            elif self.phase[i] == 1:
                self.brake_t[i] += 1
                if self.brake_t[i] > SETTLE and speed[i] > V_MIN:
                    self.reads[i].append(-o[i, ACC_X])
                if self.brake_t[i] >= BRAKE_STEPS or (self.brake_t[i] > SETTLE and speed[i] <= V_MIN):
                    self.phase[i] = 2
                    self.g_bt[i] = self.k * float(np.mean(self.reads[i])) if self.reads[i] else 0.7
                else:
                    override[i] = -1.0
            if self.lap[i] == 0:
                self.ests[i].append(est[i])
            if self.phase[i] < 2:
                g = min(est[i], 0.7)
            else:
                g = self.g_bt[i] * self.ratio(i, est[i])
            x[i, dy.OBS_DIM + 2] = g
            if self.lap[i] == 0:
                if self.phase[i] == 2:
                    k = int(self.odo[i] // sm.BIN)
                    s, c = self.sums[i].get(k, (0.0, 0))
                    self.sums[i][k] = (s + g, c + 1)
                lvl = self.g_bt[i] if self.phase[i] == 2 else min(est[i], 0.7)
                x[i, sm.MAP_COLS] = lvl * dy.DIRT_GRIP
            else:
                x[i, sm.MAP_COLS] = self.lookup(i)
                self.map_err.append(float(np.abs(x[i, sm.MAP_COLS] - o[i, sm.MAP_COLS]).mean()))
                self.odo_err.append(float(self.odo[i] - (b.progress[i] - b.length[i] * self.lap[i])))
        with torch.no_grad():
            a = np.tanh(self.net.actor(torch.as_tensor(x)).numpy())
        m = ~np.isnan(override)
        a[m, 1] = override[m]
        self.odo += np.maximum(o[:, sim.N_RAYS], 0) * sim.V_MAX * sim.DT
        return a


def calibrate(run="runs/p35-e5-lo-s0"):
    torch.set_num_threads(1)
    xs, ys = [], []
    for s in range(5):
        g, r, _ = _test(run, "train", s)
        xs += list(r)
        ys += list(g)
    xs, ys = np.array(xs), np.array(ys)
    ok = ~np.isnan(xs) & (xs > 0)
    k = float(np.sum(xs[ok] * ys[ok]) / np.sum(xs[ok] ** 2))   # least squares through the origin
    res = dict(k=k, n=int(ok.sum()), rmse_train=float(np.sqrt(np.mean((k * xs[ok] - ys[ok]) ** 2))))
    # check (not used for fitting): how well it extrapolates to the gate's wider range
    g, r, _ = _test(run, "gate", 99)
    ok = ~np.isnan(r)
    err = k * r[ok] - g[ok]
    res["gate_check"] = dict(n=int(ok.sum()), mean_abs=float(np.abs(err).mean()),
                             by_grip={f"{lo}-{hi}": float(np.abs(err[(g[ok] >= lo) & (g[ok] < hi)]).mean())
                                      for lo, hi in ((0.5, 0.7), (0.7, 0.9), (0.9, 1.1), (1.1, 1.3))})
    CAL.parent.mkdir(exist_ok=True)
    CAL.write_text(json.dumps(res, indent=1))
    print(res)


def _test(run, physics, seed, n=40):
    """Run only the brake test (the driver's own loop): returns true grip, K=1 reading, on-dirt flag."""
    net, base, est = sm.load(run)
    drv = BrakeTestDriver(net, base, est, k=1.0)
    b = dy.DynamicRacerBatch(n, tracks=list(range(500 + 100 * seed, 500 + 100 * seed + n)), start="line",
                             physics=physics, seed=seed, laps=1, max_steps=300, **sm.CAR)
    b.reset()
    for _ in range(150):
        a = drv(b)
        if (drv.phase == 2).all():
            break
        b.step(a)
    return b.grip.copy(), drv.g_bt.copy(), None


def evaluate(run, out_name="eval_braketest.json", seeds=None, phys_seed=None):
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    net, base, est = sm.load(run)
    kw = {} if seeds is None else dict(seeds=seeds)
    drv = BrakeTestDriver(net, base, est)
    nom = robust.laps(drv, outlap=True, **kw, **sm.CAR)
    diag = dict(map_err=float(np.mean(drv.map_err)) if drv.map_err else None)
    gk = dict(kw, **({} if phys_seed is None else dict(seed=phys_seed)))
    gate = robust.gate_laps(drv, outlap=True, **gk, **sm.CAR)
    res = dict(nominal=nom, gate=gate, **diag, wall_s=round(time.perf_counter() - t0))
    (Path(run) / out_name).write_text(json.dumps(res, indent=1, default=float))
    print(f"brake test: nominal {nom['completion']:.0%} {nom['mean_lap_time']}; gate range {gate['completion']:.0%}; "
          f"map err {diag['map_err']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["calibrate", "eval"])
    ap.add_argument("--run", default="runs/p35-e5-lo-s0")
    a = ap.parse_args()
    if a.cmd == "calibrate":
        calibrate(a.run)
    else:
        evaluate(a.run)


if __name__ == "__main__":
    main()
