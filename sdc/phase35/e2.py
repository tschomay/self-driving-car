"""Phase 3.5 E2: the ceiling under the RL car's constraints.

    python -m sdc.phase35.e2 --variant full|blind|blind_est [--est runs/p35-e1-s1-reliable-est]

The hand-coded driver (dy.DynamicPursuit) normally knows the dirt map, the true grip and the mass. Variants:
  full       all of that (the 25.4 s reference)
  blind      no dirt map, no mass (assumes nominal); still the true grip
  blind_est  no dirt map, no mass, and the grip from a learned 1 s window estimator (sdc/phase3/grip_estimator)
             in place of the true grip: the information the RL car drives with (plus the track shape ahead)
For each variant, a grid of safety margins (lat_margin, brake_margin) is scored on the validation set: gate-range
completion (100 laps) and nominal mean lap (20 tracks). The answer is the fastest setting with >= 90% gate range.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from sdc.phase3 import grip_estimator as ge
from sdc.phase3 import robust
from sdc.racer import dynamics as dy

LAT = (0.4, 0.5, 0.6, 0.7, 0.8)
BRAKE = (0.3, 0.4, 0.5)
MIN_GATE = 0.90


class _Belief:
    """The batch as the handicapped driver sees it: everything forwarded, except grip, mass and dirt."""

    def __init__(self, b, grip, dirt):
        self._b, self.grip, self.dirt_pts, self.mass = b, grip, dirt, np.ones(b.n)

    def __getattr__(self, k):
        return getattr(self._b, k)


class Handicapped:
    def __init__(self, variant, est=None, **margins):
        self.variant, self.est = variant, est
        self.drv = dy.DynamicPursuit(**margins)

    def __call__(self, b):
        if self.variant == "full":
            return self.drv(b)
        if self.variant == "blind":
            grip = b.grip
        else:
            with torch.no_grad():
                grip = np.clip(self.est(torch.as_tensor(ge.features(b.obs(), b))).numpy(), 0.3, 1.6)
        return self.drv(_Belief(b, grip, np.zeros_like(b.dirt_pts)))


def collect(pol, n_steps, seed):
    """Like grip_estimator.collect, but driven by the hand-coded driver: (features, true grip) per episode."""
    b = dy.DynamicRacerBatch(16, physics="train", seed=seed, **ge.car_options("window"))
    b.reset()
    rng = np.random.default_rng(seed)
    cur_x, cur_y, seqs = [[] for _ in range(b.n)], [[] for _ in range(b.n)], []
    for _ in range(n_steps // b.n):
        o = b.obs()
        f = ge.features(o, b)
        for i in range(b.n):
            cur_x[i].append(f[i])
            cur_y[i].append(o[i, b.grip_col])
        a = np.clip(pol(b) + 0.1 * rng.standard_normal((b.n, 2)), -1, 1)
        _, _, term, trunc, _ = b.step(a)
        for i in np.flatnonzero(term | trunc):
            seqs.append((np.array(cur_x[i]), np.array(cur_y[i], np.float32)))
            cur_x[i], cur_y[i] = [], []
    return seqs + [(np.array(x), np.array(y, np.float32)) for x, y in zip(cur_x, cur_y) if x]


def fit_estimator(out, rounds=4, steps=60_000, seed=0):
    """A 1 s window grip estimator trained on the hand-coded driver's own driving (DAgger: round 0 drives on
    the true grip, later rounds on the current estimate), across the margin grid."""
    torch.manual_seed(seed)
    est = ge._new("window")
    data, log = [], []
    for r in range(rounds):
        for j, (l, br) in enumerate(itertools.product(LAT, BRAKE)):
            pol = Handicapped("blind" if r == 0 else "blind_est", est, lat_margin=l, brake_margin=br)
            data += collect(pol, steps // 15, seed * 1000 + r * 100 + j)
        rmse = ge.fit(est, data, seed=seed)
        log.append(dict(round=r, samples=int(sum(len(q[1]) for q in data)), rmse=round(rmse, 4)))
        print(log[-1], flush=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(est.state_dict(), out / "estimator.pt")
    (out / "metrics.json").write_text(json.dumps(dict(rounds=log), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["full", "blind", "blind_est"])
    ap.add_argument("--fit-est", metavar="OUT", help="train an estimator on the hand-coded driver's driving")
    ap.add_argument("--est", default="runs/p35-e1-s1-reliable-est")
    ap.add_argument("--out", default="runs/p35-e2")
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.fit_est:
        return fit_estimator(a.fit_est)
    car = ge.car_options("window")
    est = None
    if a.variant == "blind_est":
        est = ge._new("window")
        est.load_state_dict(torch.load(Path(a.est) / "estimator.pt"))
    grid = [dict(lat_margin=0.7, brake_margin=0.5)] if a.variant == "full" else \
        [dict(lat_margin=l, brake_margin=br) for l, br in itertools.product(LAT, BRAKE)]
    rows = []
    for g in grid:
        pol = Handicapped(a.variant, est, **g)
        nom = robust.laps(pol, **car)
        gate = robust.gate_laps(pol, **car)
        rows.append(dict(g, completion=nom["completion"], mean_lap_time=nom["mean_lap_time"],
                         gate_range=gate["completion"]))
        print(rows[-1], flush=True)
    ok = [r for r in rows if r["gate_range"] >= MIN_GATE and r["mean_lap_time"]]
    best = min(ok, key=lambda r: r["mean_lap_time"]) if ok else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = a.variant if a.variant != "blind_est" else f"blind_est-{Path(a.est).name}"
    (out / f"{name}.json").write_text(json.dumps(dict(variant=a.variant, est=a.est, rows=rows, best=best), indent=1))
    print("best", best)


if __name__ == "__main__":
    main()
