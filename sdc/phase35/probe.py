"""Phase 3.5 E7: probe the surface repeatedly on the out-lap (owner's idea after E6).

    python -m sdc.phase35.probe calibrate                 # K for a short pulse (training-range physics only)
    python -m sdc.phase35.probe eval --run runs/p35-e5-lo-s0

Out-lap: the car's own steering; every PROBE_EVERY metres of odometer (when its steering is nearly straight) it
gives a short full-brake pulse of PULSE steps. The wheels lock; the accelerometer's readings from step SETTLE on
(while the speed reading is above V_MIN), × K, are the grip of the surface under it. If the speed reading is
below V_PROBE when a probe is due, the throttle is overridden to full until it gets there (only while steering
is nearly straight). The readings fill a map in 5 m odometer bins (median per bin; empty bins interpolated).
While mapping, the car's grip input is its latest reading and its map input is the median reading so far ×
DIRT_GRIP (cautious).
Timed lap: grip input = map at the current odometer position; map input = map 5–40 m ahead (as in E5/E6). No
window estimator is used.
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

PROBE_EVERY = 6.0   # m of odometer between probes
PULSE = 4           # steps of full brake
SETTLE = 2          # readings count from this step of the pulse (after the delay and the lock)
V_PROBE = 6.0       # m/s: speed up to this before a probe
V_MIN = 1.5
STEER_OK = 0.35     # |steering command| below which a probe or a throttle override is allowed
START_GRIP = 0.7
CAL = Path("runs/p35-e7-calibration.json")
ACC_X = dy.OBS_DIM


class ProbeDriver(sm.SelfMapDriver):
    def __init__(self, net, base, k=None, record_truth=False):
        super().__init__(net, base, est=None)
        self.k = k if k is not None else json.loads(CAL.read_text())["k"]
        self.record_truth = record_truth

    def reset_state(self, n):
        super().reset_state(n)
        self.pulse_t = np.zeros(n, int)       # 0 = not probing
        self.next_probe = np.full(n, 2.0)
        self.reads = [[] for _ in range(n)]
        self.pulse_odo = np.zeros(n)
        self.pulse_truth = np.zeros(n)
        self.readings = [[] for _ in range(n)]     # (odo, grip reading)
        self.latest = np.full(n, START_GRIP)
        self.truth_pairs = []                      # (reading with K, true grip) for calibration / checks
        self.n_probes = np.zeros(n, int)
        self._tab = {}

    def table(self, i):
        L = max(self.out_len[i], sm.BIN)
        nb = int(np.ceil(L / sm.BIN))
        bins = [[] for _ in range(nb)]
        for odo, g in self.readings[i]:
            bins[int(odo // sm.BIN) % nb].append(g)
        tab = np.array([np.median(v) if v else np.nan for v in bins])
        if np.isnan(tab).all():
            tab[:] = START_GRIP
        idx = np.arange(nb)
        good = ~np.isnan(tab)
        return np.interp(idx, idx[good], tab[good], period=nb), L, nb

    def __call__(self, b):
        if self.n != b.n or (b.steps == 0).all():
            self.reset_state(b.n)
        o = b.obs()
        lap_now = np.floor(b.progress / b.length + 1e-9).astype(int)
        newly = np.flatnonzero(lap_now > self.lap)
        for i in newly:
            if self.lap[i] == 0:
                self.out_len[i] = self.odo[i]
                self._tab[i] = self.table(i)
            self.odo[i] = 0.0
        self.lap = np.maximum(self.lap, lap_now)
        x = o[:, :self.base].copy()
        speed = o[:, sim.N_RAYS] * sim.V_MAX
        for i in range(b.n):
            if self.lap[i] == 0:
                med = np.median([g for _, g in self.readings[i]]) if self.readings[i] else START_GRIP
                x[i, dy.OBS_DIM + 2] = self.latest[i]
                x[i, sm.MAP_COLS] = med * dy.DIRT_GRIP
            else:
                tab, L, nb = self._tab[i]
                q = ((self.odo[i] + np.array((0.0,) + dy.GRIPMAP_AHEAD)) % L) // sm.BIN
                v = tab[q.astype(int) % nb]
                x[i, dy.OBS_DIM + 2] = v[0]
                x[i, sm.MAP_COLS] = v[1:]
                self.map_err.append(float(np.abs(v[1:] - o[i, sm.MAP_COLS]).mean()))
        with torch.no_grad():
            a = np.tanh(self.net.actor(torch.as_tensor(x)).numpy())
        for i in range(b.n):
            if self.lap[i] != 0:
                continue
            straight = abs(a[i, 0]) < STEER_OK
            if self.pulse_t[i] == 0 and self.odo[i] >= self.next_probe[i]:
                if speed[i] >= V_PROBE and straight:
                    self.pulse_t[i] = 1
                    self.reads[i] = []
                    self.pulse_odo[i] = self.odo[i]
                    self.pulse_truth[i] = o[i, b.grip_col] if b.grip_col is not None else np.nan
                elif straight:
                    a[i, 1] = 1.0
            if self.pulse_t[i] > 0:
                a[i, 1] = -1.0
                if self.pulse_t[i] > SETTLE and speed[i] > V_MIN:
                    self.reads[i].append(-o[i, ACC_X])
                self.pulse_t[i] += 1
                if self.pulse_t[i] > PULSE:
                    self.pulse_t[i] = 0
                    self.next_probe[i] = self.odo[i] + PROBE_EVERY
                    if self.reads[i]:
                        g = self.k * float(np.mean(self.reads[i]))
                        self.latest[i] = g
                        self.readings[i].append((self.pulse_odo[i], g))
                        self.n_probes[i] += 1
                        self.truth_pairs.append((g, float(self.pulse_truth[i])))
        self.odo += np.maximum(o[:, sim.N_RAYS], 0) * sim.V_MAX * sim.DT
        return a


def calibrate(run="runs/p35-e5-lo-s0"):
    """K: least squares (through 0) of true surface grip on the K=1 pulse reading, training-range physics only;
    then a check on gate-range physics (not used for fitting)."""
    torch.set_num_threads(1)
    net, base = sm.load_net(run)

    def pairs(physics, seed):
        drv = ProbeDriver(net, base, k=1.0)
        robust.gate_laps(drv, physics=physics, seed=seed, draws=1, outlap=True, **sm.CAR)
        return np.array(drv.truth_pairs)

    P = np.concatenate([pairs("train", 4242 + s) for s in range(2)])
    k = float(np.sum(P[:, 0] * P[:, 1]) / np.sum(P[:, 0] ** 2))
    res = dict(k=k, n=len(P), rmse_train=float(np.sqrt(np.mean((k * P[:, 0] - P[:, 1]) ** 2))))
    Q = pairs("gate", 777)
    err = k * Q[:, 0] - Q[:, 1]
    res["gate_check"] = dict(n=len(Q), mean_abs=float(np.abs(err).mean()), bias=float(err.mean()),
                             dirt_mean=float((k * Q[Q[:, 1] < 0.9 * np.median(Q[:, 1]), 0]).mean()))
    CAL.write_text(json.dumps(res, indent=1))
    print(res)


def evaluate(run, out_name="eval_probe.json", seeds=None, phys_seed=None):
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    net, base = sm.load_net(run)
    kw = {} if seeds is None else dict(seeds=seeds)
    drv = ProbeDriver(net, base)
    nom = robust.laps(drv, outlap=True, **kw, **sm.CAR)
    diag = dict(map_err=float(np.mean(drv.map_err)) if drv.map_err else None,
                probes_per_lap=float(np.mean(drv.n_probes)))
    gk = dict(kw, **({} if phys_seed is None else dict(seed=phys_seed)))
    gate = robust.gate_laps(drv, outlap=True, **gk, **sm.CAR)
    res = dict(nominal=nom, gate=gate, **diag, wall_s=round(time.perf_counter() - t0))
    (Path(run) / out_name).write_text(json.dumps(res, indent=1, default=float))
    print(f"probes: nominal {nom['completion']:.0%} {nom['mean_lap_time']}; gate range {gate['completion']:.0%}; "
          f"map err {diag['map_err']}; probes/out-lap {diag['probes_per_lap']:.0f}", flush=True)
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
