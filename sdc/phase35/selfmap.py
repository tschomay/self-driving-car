"""Phase 3.5 E5: the car builds its own grip map on an out-lap and drives on it (no oracle inputs).

    python -m sdc.phase35.selfmap fit  --policy runs/p35-e4-lo-s0-fast --out runs/p35-e5-lo-s0   # grip estimator
    python -m sdc.phase35.selfmap eval --run runs/p35-e5-lo-s0                                      # validation

What the car knows (owner's rules, 2026-11-11): its own sensors, plus a signal when it crosses the start line.
  - grip: a 1 s window estimator (sdc/phase3/grip_estimator.py) replaces the true grip, trained on this car's
    own driving (DAgger: round 0 drives on the true grip, later rounds on the estimate).
  - position: an odometer, the noisy speed reading added up each step, reset at each line crossing.
  - out-lap: the car records its grip estimate in 5 m bins of odometer distance. Its map input meanwhile says
    "everything ahead may be dirt": the estimate's running mean × DIRT_GRIP (cautious; see E3's ablation).
  - timed lap: the map input is the recorded grip at odometer + 5, 10, ..., 40 m (wrapping at the out-lap's
    length).
Diagnostics: map error (|map input − true gripmap|) on the timed lap, and the odometer's error vs. the true
distance along the centerline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from sdc.phase1 import my_ppo as owner
from sdc.phase3 import grip_estimator as ge
from sdc.phase3 import robust
from sdc.racer import dynamics as dy
from sdc.racer import sim

BIN = 5.0
CAR = dict(realism=3, sensors=("accel", "grip", "gripmap"), history=ge.HISTORY)
MAP_COLS = slice(dy.OBS_DIM + 3, dy.OBS_DIM + 3 + len(dy.GRIPMAP_AHEAD))


def load_net(policy_dir):
    probe = dy.DynamicRacerBatch(1, realism=3, sensors=CAR["sensors"])
    net = owner.ActorCritic(obs_dim=probe.obs_dim)
    net.load_state_dict(torch.load(Path(policy_dir) / "model.pt"))
    return net, probe.obs_dim


class SelfMapDriver:
    """policy(b) for robust.laps(outlap=True). grip and map from the car's own readings (or the truth, for the
    estimator's round 0)."""

    def __init__(self, net, base_dim, est, true_grip=False, true_map=False):
        self.net, self.base, self.est = net, base_dim, est
        self.true_grip, self.true_map = true_grip, true_map
        self.reset_state(0)

    def reset_state(self, n):
        self.n = n
        self.odo = np.zeros(n)
        self.lap = np.zeros(n, int)
        self.sums = [dict() for _ in range(n)]
        self.out_len = np.zeros(n)
        self.est_sum = np.zeros(n)
        self.est_n = np.zeros(n)
        self.map_err, self.odo_err = [], []

    def grip_estimate(self, o, b):
        with torch.no_grad():
            return np.clip(self.est(torch.as_tensor(ge.features(o, b))).numpy(), 0.2, 1.8)

    def __call__(self, b):
        if self.n != b.n or (b.steps == 0).all():
            self.reset_state(b.n)
        o = b.obs()
        fresh = b.steps == 0
        if fresh.any():   # a car starting a new episode (training-style batches)
            for i in np.flatnonzero(fresh):
                self.odo[i], self.lap[i], self.sums[i] = 0.0, 0, {}
                self.est_sum[i] = self.est_n[i] = 0.0
        lap_now = np.floor(b.progress / b.length + 1e-9).astype(int)
        crossed = lap_now > self.lap
        for i in np.flatnonzero(crossed):
            if self.lap[i] == 0:
                self.out_len[i] = self.odo[i]
            self.odo[i] = 0.0
        self.lap = np.maximum(self.lap, lap_now)
        g = o[:, dy.OBS_DIM + 2].copy() if self.true_grip else self.grip_estimate(o, b)
        x = o[:, :self.base].copy()
        x[:, dy.OBS_DIM + 2] = g
        out_lap = self.lap == 0
        self.est_sum[out_lap] += g[out_lap]
        self.est_n[out_lap] += 1
        for i in range(b.n):
            if out_lap[i]:   # record and drive cautiously
                k = int(self.odo[i] // BIN)
                s, c = self.sums[i].get(k, (0.0, 0))
                self.sums[i][k] = (s + g[i], c + 1)
                if not self.true_map:
                    x[i, MAP_COLS] = (self.est_sum[i] / self.est_n[i]) * dy.DIRT_GRIP
            elif not self.true_map:
                x[i, MAP_COLS] = self.lookup(i)
                self.map_err.append(float(np.abs(x[i, MAP_COLS] - o[i, MAP_COLS]).mean()))
                self.odo_err.append(float(self.odo[i] - (b.progress[i] - b.length[i] * self.lap[i])))
        with torch.no_grad():
            a = np.tanh(self.net.actor(torch.as_tensor(x)).numpy())
        self.odo += np.maximum(o[:, sim.N_RAYS], 0) * sim.V_MAX * sim.DT   # noisy speed reading × dt
        return a

    def lookup(self, i):
        L = max(self.out_len[i], BIN)
        nb = int(np.ceil(L / BIN))
        tab = np.array([self.sums[i][k][0] / self.sums[i][k][1] if k in self.sums[i] else np.nan for k in range(nb)])
        if np.isnan(tab).all():
            tab[:] = 1.0
        idx = np.arange(nb)
        good = ~np.isnan(tab)
        tab = np.interp(idx, idx[good], tab[good], period=nb)
        q = ((self.odo[i] + np.array(dy.GRIPMAP_AHEAD)) % L) // BIN
        return tab[q.astype(int) % nb]


def collect(driver, n_steps, seed):
    """(features, true grip) per episode, driving training-range cars on training tracks (two laps each)."""
    b = dy.DynamicRacerBatch(16, physics="train", seed=seed, tracks="random", laps=2, max_steps=2400, **CAR)
    b.reset()
    driver.reset_state(b.n)
    rng = np.random.default_rng(seed)
    cur_x, cur_y, seqs = [[] for _ in range(b.n)], [[] for _ in range(b.n)], []
    for _ in range(n_steps // b.n):
        o = b.obs()
        f = ge.features(o, b)
        for i in range(b.n):
            cur_x[i].append(f[i])
            cur_y[i].append(o[i, b.grip_col])
        a = np.clip(driver(b) + 0.1 * rng.standard_normal((b.n, 2)), -1, 1)
        _, _, term, trunc, _ = b.step(a)
        for i in np.flatnonzero(term | trunc):
            seqs.append((np.array(cur_x[i]), np.array(cur_y[i], np.float32)))
            cur_x[i], cur_y[i] = [], []
    return seqs + [(np.array(x), np.array(y, np.float32)) for x, y in zip(cur_x, cur_y) if x]


def fit(policy, out, seed=0, rounds=4, steps=60_000):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    net, base = load_net(policy)
    est = ge._new("window")
    data, log = [], []
    for r in range(rounds):
        drv = SelfMapDriver(net, base, est, true_grip=(r == 0), true_map=(r == 0))
        data += collect(drv, steps, seed * 100 + r)
        rmse = ge.fit(est, data, seed=seed)
        log.append(dict(round=r, samples=int(sum(len(q[1]) for q in data)), rmse=round(rmse, 4)))
        print(log[-1], flush=True)
    torch.save(est.state_dict(), out / "estimator.pt")
    shutil.copy(Path(policy) / "model.pt", out / "model.pt")
    (out / "metrics.json").write_text(json.dumps(dict(policy=str(policy), seed=seed, rounds=log), indent=1))


def load(run):
    run = Path(run)
    net, base = load_net(run)
    est = ge._new("window")
    est.load_state_dict(torch.load(run / "estimator.pt"))
    return net, base, est


def evaluate(run, seeds=None, phys_seed=None, out_name="eval.json"):
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    net, base, est = load(run)
    kw = {} if seeds is None else dict(seeds=seeds)
    res = {}
    for name, true in (("self", False), ("oracle", True)):
        drv = SelfMapDriver(net, base, est, true_grip=true, true_map=true)
        nom = robust.laps(drv, outlap=True, **kw, **CAR)
        diag = dict(map_err=float(np.mean(drv.map_err)) if drv.map_err else None,
                    odo_err_abs=float(np.mean(np.abs(drv.odo_err))) if drv.odo_err else None)
        gk = dict(kw, **({} if phys_seed is None else dict(seed=phys_seed)))
        gate = robust.gate_laps(drv, outlap=True, **gk, **CAR)
        res[name] = dict(nominal=nom, gate=gate, **diag)
        print(name, f"nominal {nom['completion']:.0%} {nom['mean_lap_time']}; gate range {gate['completion']:.0%}; "
              f"map err {diag['map_err']}, odometer err {diag['odo_err_abs']}", flush=True)
    res["wall_s"] = round(time.perf_counter() - t0)
    (Path(run) / out_name).write_text(json.dumps(res, indent=1, default=float))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fit", "eval"])
    ap.add_argument("--policy")
    ap.add_argument("--out")
    ap.add_argument("--run")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(a.policy, a.out, a.seed)
    else:
        evaluate(a.run)


if __name__ == "__main__":
    main()
