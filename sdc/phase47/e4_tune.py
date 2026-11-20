"""Phase 4.7 E4, step 1: re-tune the teacher where it fails (landings), then the student copies it.

    python -m sdc.phase47.e4_tune [--procs 2]   -> runs/p47-e4-tune.json, runs/p47-e4-teacher.json

The teacher's failures are nearly all landings (61 of 65 in 5 passes): rollovers on touchdown and fish-tails in the
3 s after, mostly on doubles taken 1.6-2 m/s under their design speed. Candidate settings over teacher.TRUGGY:
approach speed into doubles (jump_v), yaw-rate damping (k_rate), and the course plan (each jump's design speed).
Each drives TUNE_LOOPS practice loops from the train split (not the selection loops E4 is scored on) x 3 starts,
gate range, REPEATS passes with the same tiny action noise as where.py. The best mean wins; the current teacher
keeps its place unless another setting beats it by more than MARGIN finishes a pass.
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

OUT = Path("runs/p47-e4-tune.json")
TEACHER = Path("runs/p47-e4-teacher.json")
TUNE_LOOPS, REPEATS, MARGIN = 50, 3, 2.0
SETTINGS = {
    "current": {},
    "double 7.4": dict(jump_v={"double": 7.4}),
    "double 8.0": dict(jump_v={"double": 8.0}),
    "double 8.6": dict(jump_v={"double": 8.6}),
    "k_rate 0.5": dict(k_rate=0.5),
    "double 8.0, k_rate 0.5": dict(jump_v={"double": 8.0}, k_rate=0.5),
    "design speeds": dict(plan=True),
}


def make(over, seed):
    import os
    return Noisy(LoopExpert(**{**TRUGGY, **over}), seed * 1000 + os.getpid())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    courses = json.loads(e2_prep.OUT.read_text())["split"]["train"][:TUNE_LOOPS]
    env = {k: v for k, v in env_kw(val=True).items() if k != "expo"}
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, over in SETTINGS.items():
        if name in out:
            continue
        fin = []
        for r in range(REPEATS):
            eps = cm.prun(functools.partial(make, over, r), courses, procs=a.procs, **env)
            fin.append(sum(bool(e["finished"]) for e in eps))
        laps = [e["l"] * 0.02 for e in eps if e["finished"]]
        out[name] = dict(over=over, finished_per_pass=fin, mean=float(np.mean(fin)), lap_s=float(np.median(laps)))
        OUT.write_text(json.dumps(out, indent=1))
        print(name, out[name], flush=True)
    best = max(out, key=lambda k: out[k]["mean"])
    if out[best]["mean"] - out["current"]["mean"] <= MARGIN:
        best = "current"
    TEACHER.write_text(json.dumps(dict(out[best]["over"], _name=best)))
    print("chosen:", best, out[best], flush=True)


if __name__ == "__main__":
    main()
