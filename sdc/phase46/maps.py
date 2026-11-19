"""Phase 4.6 E3: out-lap maps of the turns.

    python -m sdc.phase46.maps            # practice loops (training range) and validation loops (gate range, 3 starts)

For every run (loop, start), the remote-driven stand-in (teacher.OutLapDriver) drives one slow lap at that run's
physics while the car records its parts-list sensors. The owner's dead reckoning turns the log into a path, closes
it at the finish line, and the path's heading change per metre is the map: {"loop|x0": {"ds", "kappa", "length",
"reason"}}. An out-lap that didn't finish leaves an empty map (the timed lap then runs without one).
Writes runs/p46-maps-train.json and runs/p46-maps-val.json.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sdc.jumps import driver as dr
from sdc.phase46 import common as cm
from sdc.phase46 import laps
from sdc.phase46 import my_deadreckoning as mdr
from sdc.phase46.teacher import OUTLAP_TRUGGY, OutLapDriver

DS = 0.25


def make_map(log):
    """The map of the turns from one out-lap's sensor log (no truth used)."""
    if log["reason"] != "finished":
        return dict(ds=DS, kappa=[], length=0.0, reason=log["reason"])
    o = log["obs"]
    est = mdr.DeadReckoning()
    est.reset(log["still_gyro"])
    for i in range(1, len(o)):
        att = o[i, laps.I_ATT] * 0.5
        est.update(o[i, laps.I_GYRO] * 5.0, o[i, laps.I_SPEED] * 10.0, att[0], att[1], acc=o[i, laps.I_ACC] * 9.81,
                   steer=o[i, laps.I_ACT][0])
    marks = kickers(o[:, laps.I_ATT.start] * 0.5, np.linalg.norm(o[:, laps.I_ACC], axis=1), np.array(est.path)[:, 0])
    path, _ = est.at_line()
    # the path starts START_X past the line and ends at it: one full lap of odometer distance, indexed from the line
    s = path[:, 0] - path[0, 0] + est.start_x
    lap = float(path[-1, 0] - path[0, 0]) + est.start_x
    grid = np.arange(0.0, lap, DS)
    h = np.interp(grid, s, path[:, 3], left=path[0, 3])
    k = np.gradient(h, grid)
    n = int(1.0 / DS)
    k = np.convolve(np.concatenate([k[-n:], k, k[:n]]), np.ones(2 * n + 1) / (2 * n + 1), mode="valid")
    return dict(ds=DS, kappa=[round(float(x), 4) for x in k], length=round(lap, 2), reason="finished",
                kickers=[round(float(m), 2) for m in marks])


KICK_PITCH = np.deg2rad(12)     # the filter's pitch climbing through this = the car is on a kicker
KICK_STEPS = 3


def kickers(pitch, acc_g, odo, gap=8.0):
    """Landmarks for re-syncing the odometer: where the car starts climbing a kicker (the attitude filter's pitch
    above KICK_PITCH for KICK_STEPS steps, on the ground). Only the first within `gap` metres is kept: on a double
    the slow lap climbs both rollers, the race lap flies over the second."""
    marks, run = [], 0
    for i in range(len(pitch)):
        run = run + 1 if (pitch[i] > KICK_PITCH and acc_g[i] > 0.4) else 0
        if run == KICK_STEPS and (not marks or odo[i] - marks[-1] > gap):
            marks.append(odo[i])
    return marks


def _job(args):
    course, x0, d, phys_seed, real = (*args, "all")[:5] if len(args) == 4 else args
    import torch
    torch.set_num_threads(1)
    drv = OutLapDriver(**(OUTLAP_TRUGGY if real == "truggy" else {}))
    log = laps.record(drv, course, x0, dr=d, phys_seed=phys_seed, max_steps=6000, real=real)
    return f"{course}|{x0:.2f}", make_map(log)


def collect(courses, starts, d, phys_seed=0, procs=4, real="all"):
    with Pool(procs) as p:
        return dict(p.map(_job, [(c, x0, d, phys_seed, real) for c in courses for x0 in starts], chunksize=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    for name, courses, starts, d in (("val", cm.VAL_SEEDS, dr.STARTS, "gate"), ("train", cm.PRACTICE, (0.0,), "train")):
        maps = collect(courses, starts, d, procs=a.procs)
        Path(f"runs/p46-maps-{name}.json").write_text(json.dumps(maps))
        print(name, len(maps), "maps;", sum(m["reason"] == "finished" for m in maps.values()), "out-laps finished", flush=True)


if __name__ == "__main__":
    main()
