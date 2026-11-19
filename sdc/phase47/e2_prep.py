"""Phase 4.7 E2 prep: split the practice loops, calibrate drill strengths, and score the base car on the exam.

    python -m sdc.phase47.e2_prep      # -> runs/p47-e2-prep.json

Practice loops whose out-lap finished (as bc_loop uses) are split, in order: TRAIN (harvested by training runs),
SELECT (50 loops x 3 starts, gate range: checkpoint picks), CALIB (30: calibration drills), EXAM (30: the exam).
"""

from __future__ import annotations

import functools
import multiprocessing as mp
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from sdc.phase46 import common as cm
from sdc.phase46.bc_loop import TRAIN_MAP
from sdc.phase46.teacher import TRUGGY, LoopExpert
from sdc.phase47 import drills2 as d2
from sdc.phase47.train_residual import BASE, env_kw

OUT = Path("runs/p47-e2-prep.json")
CALIB = Path("runs/p47-e2-calib.json")
SELECT_REF = Path("runs/p47-e2-select-teacher.pkl")   # the teacher on the selection runs (time limit)
EXAM_SEED = 4701


def split():
    maps = json.loads(Path(TRAIN_MAP).read_text())
    loops = [c for c in cm.PRACTICE if maps[f"{c}|0.00"]["kappa"]]
    return dict(train=loops[:-110], select=loops[-110:-60], calib=loops[-60:-30], exam=loops[-30:])


_POOL = None


def _calibrate(kind):
    torch.set_num_threads(1)
    return kind, d2.calibrate_kind(_POOL, kind, BASE, env_kw(), log=lambda *a: print(*a, flush=True))


def exam_set(amps):
    sp = split()
    pool, hv = d2.harvest(BASE, sp["exam"], env_kw(), seed=EXAM_SEED)
    return d2.exam_starts(pool, amps), hv


def main():
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    sp = split()
    print({k: len(v) for k, v in sp.items()}, flush=True)
    env = {k: v for k, v in env_kw(val=True).items() if k != "expo"}
    ref = pickle.load(open(SELECT_REF, "rb")) if SELECT_REF.exists() else cm.prun(functools.partial(LoopExpert, **TRUGGY), sp["select"], procs=4, **env)
    if not SELECT_REF.exists():
        pickle.dump(ref, open(SELECT_REF, "wb"))
    print("teacher on selection runs", sum(e["finished"] for e in ref), "/", len(ref), flush=True)
    if CALIB.exists():   # a finished calibration (e.g. before a crash later in prep)
        cal, hv = json.loads(CALIB.read_text()), None
    else:
        pool, hv = d2.harvest(BASE, sp["calib"], env_kw(), seed=4700)
        print("calib harvest", hv, flush=True)
        global _POOL
        _POOL = pool
        with mp.get_context("fork").Pool(len(d2.TUNED)) as p:    # one kind per process (the pool is inherited)
            cal = dict(p.map(_calibrate, d2.TUNED))
        CALIB.write_text(json.dumps(cal, indent=1))
    amps = {k: v["amp"] for k, v in cal.items()}
    starts, ehv = exam_set(amps)
    print("exam harvest", ehv, flush=True)
    crashed = d2.run_starts(starts, BASE, env_kw())
    ex = d2.exam_score(starts, crashed)
    print("base exam", ex, flush=True)
    d2.save_json(OUT, dict(split={k: [int(c) for c in v] for k, v in sp.items()}, calib_harvest=hv, calibration=cal,
                           amps=amps, exam_harvest=ehv, base_exam=ex, base_exam_crashed=crashed,
                           wall_s=round(time.perf_counter() - t0)))


if __name__ == "__main__":
    main()
