"""Phase 4.7, stepping back: how much faster could a driver that knows the track go, and at what cost in finishes?

    python -m sdc.phase47.headroom [--procs 4]   -> runs/p47-headroom.json

The re-tuned teacher (TRUGGY + doubles at 7.4 m/s) plans for 2.5 m/s^2 of cornering and cruises at 7.4 m/s, on a
motor with a 16 m/s no-load top speed and tires with roughly 5-12 m/s^2 of grip (gate range). Every car so far laps
in about 30.5-31 s. This sweeps the teacher's aggression (cornering, braking, cruise speed off the jumps; jump
approach speeds unchanged) on 50 practice loops from the train split x 3 starts, gate range, with where.py's tiny
action noise, REPEATS passes each. Result: finishes per pass and the median lap difference to the baseline on runs
both finished: a rough speed/reliability frontier for a driver with perfect knowledge of the track.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import numpy as np

from sdc.phase46 import common as cm
from sdc.phase46.teacher import TRUGGY, LoopExpert
from sdc.phase47 import e2_prep
from sdc.phase47.lap_chaos import Noisy
from sdc.phase47.train_residual import env_kw

OUT = Path("runs/p47-headroom.json")
LOOPS, REPEATS = 50, 2
BASE = dict(jump_v={"double": 7.4})            # E4's re-tuned teacher
SETTINGS = {
    "teacher (E4)": {},
    "cruise 9": dict(v_cruise=9.0),
    "cruise 11": dict(v_cruise=11.0),
    "corners 3.5": dict(a_lat=3.5, a_brake=3.5),
    "corners 4.5": dict(a_lat=4.5, a_brake=4.5),
    "corners 6": dict(a_lat=6.0, a_brake=6.0),
    "corners 3.5, cruise 9": dict(a_lat=3.5, a_brake=3.5, v_cruise=9.0),
    "corners 4.5, cruise 11": dict(a_lat=4.5, a_brake=4.5, v_cruise=11.0),
    "corners 6, cruise 13": dict(a_lat=6.0, a_brake=6.0, v_cruise=13.0),
}


def make(over, seed):
    import os
    return Noisy(LoopExpert(**{**TRUGGY, **BASE, **over}), seed * 1000 + os.getpid())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    courses = json.loads(e2_prep.OUT.read_text())["split"]["train"][:LOOPS]
    env = {k: v for k, v in env_kw(val=True).items() if k != "expo"}
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, over in SETTINGS.items():
        if name in out:
            continue
        fin, laps = [], []
        for r in range(REPEATS):
            eps = cm.prun(functools.partial(make, over, r), courses, procs=a.procs, **env)
            fin.append(sum(bool(e["finished"]) for e in eps))
            laps.append([e["l"] * 0.02 if e["finished"] else None for e in eps])
        out[name] = dict(over=over, finished_per_pass=fin, mean=float(np.mean(fin)), laps=laps,
                         lap_s=float(np.median([x for p in laps for x in p if x])))
        ref = out[next(iter(SETTINGS))]["laps"]
        d = [x - y for p, q in zip(laps, ref) for x, y in zip(p, q) if x and y]
        out[name]["vs_teacher_s"] = round(float(np.median(d)), 2) if d else None
        OUT.write_text(json.dumps(out))
        print(name, {k: v for k, v in out[name].items() if k != "laps"}, flush=True)


if __name__ == "__main__":
    main()
