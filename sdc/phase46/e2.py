"""Phase 4.6 E2: the owner's dead reckoning. How far does it drift over one lap?

    python -m sdc.phase46.e2 --out runs/p46-e2

Two kinds of lap on the validation loops (20 loops x 3 starts, gate-range physics, parts-list sensors):
  slow   the out-lap: the remote-driven stand-in at 2 m/s (teacher.OutLapDriver)
  race   the teacher at race speed, three jumps (teacher.LoopExpert)
Each lap's sensor log is replayed through the owner's estimator and through its alternatives (one choice changed at
a time). Scored against the true pose, for laps that finished:
  closing error   distance between the estimated and the true position when the car is back at the line, before
                  the finish-line correction; also as a share of the distance driven
  heading error   at the line, degrees;  distance error: estimated vs true distance driven, %
  what hurts      the closing error with the true heading and the estimated distance ("distance only"), and with
                  the estimated heading and the true distance ("heading only")
  the map         after the finish-line correction: mean position error along the lap, and for each turn, how far
                  the mapped turn is from the true one (metres along the lap) and the error in its tightest radius
Writes <out>/e2.json and the logs of loop 10000 (for the report's figure).
"""

from __future__ import annotations

import argparse
import json
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sdc.jumps import loop as lp
from sdc.phase46 import common as cm
from sdc.phase46 import laps
from sdc.phase46 import my_deadreckoning as mdr
from sdc.phase46.e1 import corners
from sdc.phase46.teacher import LoopExpert, OutLapDriver

VARIANTS = {
    "owner's (gyro, tilt-corrected, hold in the air)": {},
    "no tilt correction": dict(tilt_correct=False),
    "distance: raw motor speed": dict(distance="raw"),
    "heading: steering model": dict(heading="steering"),
    "heading: blend": dict(heading="blend"),
}


WALK_SCALE = 1.0
WALK_DEG = 0.01     # °/s per sqrt(s): sdc/jumps/real.py's value (0.05 until 2026-11-17; --walk-scale 5 reproduces it)


def _record(args):
    kind, course, x0 = args
    import torch
    torch.set_num_threads(1)
    from sdc.jumps import real as rl
    rl.GYRO_WALK = np.deg2rad(WALK_DEG) * WALK_SCALE   # --walk-scale: a follow-up check on the gyro's bias drift
    pol = OutLapDriver() if kind == "slow" else LoopExpert()
    return kind, laps.record(pol, course, x0, dr="gate", phys_seed=cm.PHYS_SEED_VAL, max_steps=6000)


def _set_walk(scale):
    global WALK_SCALE
    WALK_SCALE = scale


def replay(log, **kw):
    """Run an estimator over a lap's log. Returns (estimator, (n, 3) x, y, heading per step)."""
    o = log["obs"]
    est = mdr.DeadReckoning(**kw)
    est.reset(log["still_gyro"])
    out = [(est.x, est.y, est.h)]
    for i in range(1, len(o)):
        att = o[i, laps.I_ATT] * 0.5
        out.append(est.update(o[i, laps.I_GYRO] * 5.0, o[i, laps.I_SPEED] * 10.0, att[0], att[1],
                              acc=o[i, laps.I_ACC] * 9.81, steer=o[i, laps.I_ACT][0]))
    return est, np.array(out)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def score(log, **kw):
    t = log["truth"]                       # s, d, x, y, heading, v
    est, e = replay(log, **kw)
    step = np.hypot(np.diff(t[:, 2]), np.diff(t[:, 3]))
    driven = float(step.sum())
    ds_est = np.hypot(np.diff(e[:, 0]), np.diff(e[:, 1]))
    out = dict(closing=float(np.hypot(e[-1, 0] - t[-1, 2], e[-1, 1] - t[-1, 3])), driven=driven,
               heading_deg=float(np.degrees(_wrap(e[-1, 2] - t[-1, 4]))),
               distance_pct=float(100 * (ds_est.sum() - driven) / driven))
    # what hurts: one source of error at a time
    for name, ds, h in (("distance_only", ds_est, t[1:, 4]), ("heading_only", step, e[1:, 2])):
        x = t[0, 2] + np.cumsum(ds * np.cos(h))
        y = t[0, 3] + np.cumsum(ds * np.sin(h))
        out[name] = float(np.hypot(x[-1] - t[-1, 2], y[-1] - t[-1, 3]))
    out["mean_err_raw"] = float(np.hypot(e[:, 0] - t[:, 2], e[:, 1] - t[:, 3]).mean())
    path, _ = est.at_line()
    n = min(len(path), len(t))
    out["mean_err_closed"] = float(np.hypot(path[:n, 1] - t[:n, 2], path[:n, 2] - t[:n, 3]).mean())
    # the map of the turns: each true turn's tightest point vs the map's, searched within 10 m
    c = lp.make(log["course"])
    grid, kap = mdr.curvature(path)
    turns = []
    for s0, s1, radius in corners(c):
        i0, i1 = int(s0 / c.ds), int(s1 / c.ds)
        sp = (i0 + int(np.argmax(np.abs(c.kappa[i0:i1])))) * c.ds
        m = (grid > sp - 10) & (grid < sp + 10)
        if not m.any():
            continue
        k = int(np.argmax(np.abs(kap[m])))
        turns.append(dict(where=float(grid[m][k] - sp), radius_true=radius,
                          radius_map=float(1 / max(abs(kap[m][k]), 1e-3)),
                          same_side=bool(np.sign(kap[m][k]) == np.sign(c.kappa[int(sp / c.ds)]))))
    out["turns"] = turns
    return out, e, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/p46-e2")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--walk-scale", type=float, default=1.0, help="scale the gyro's bias drift (follow-up check)")
    a = ap.parse_args()
    global WALK_SCALE
    WALK_SCALE = a.walk_scale
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, c, x0) for k in ("slow", "race") for c in cm.VAL_SEEDS for x0 in (0.0, -0.3, 0.3)]
    with Pool(a.procs, initializer=_set_walk, initargs=(a.walk_scale,)) as p:
        logs = p.map(_record, jobs, chunksize=1)
    res = {}
    keep = {}
    for kind in ("slow", "race"):
        mine = [l for k, l in logs if k == kind]
        res[kind] = dict(runs=len(mine), finished=sum(l["reason"] == "finished" for l in mine),
                         lap_s=float(np.median([len(l["obs"]) for l in mine if l["reason"] == "finished"]) * 0.02),
                         variants={})
        for name, kw in VARIANTS.items():
            rows = []
            for l in mine:
                if l["reason"] != "finished":
                    continue
                sc, e, path = score(l, **kw)
                rows.append(dict(course=l["course"], x0=l["x0"], **sc))
                if l["course"] == 10000 and l["x0"] == 0.0 and not kw:
                    keep[kind] = dict(truth=l["truth"], est=e, closed=path)
            res[kind]["variants"][name] = rows
            med = lambda k: float(np.median([r[k] for r in rows]))  # noqa: E731
            print(f"{kind:5s} {name:48s} closing {med('closing'):5.2f} m ({100 * np.median([r['closing'] / r['driven'] for r in rows]):.1f}%)"
                  f"  heading {np.median([abs(r['heading_deg']) for r in rows]):4.1f} deg  distance {med('distance_pct'):+5.1f}%"
                  f"  dist-only {med('distance_only'):5.2f}  head-only {med('heading_only'):5.2f}", flush=True)
    (out / "e2.json").write_text(json.dumps(res, default=float))
    pickle.dump(keep, open(out / "paths-10000.pkl", "wb"))


if __name__ == "__main__":
    main()
