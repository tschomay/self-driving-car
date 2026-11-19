"""Phase 4.7: an unbiased second look at chosen checkpoints, on fresh loops (47000-47033 x 3 starts, gate range).

    python -m sdc.phase47.check --out runs/p47-e1-check.json --cars base teacher runs/p47-e1-s0/ckpt/401408.pt ...

Validation picks the best of several checkpoints, which flatters the winner; these loops were never used to train,
choose or tune anything. Each car is scored on the same 102 runs; laps timed against the teacher on the same run.
"""

from __future__ import annotations

import argparse
import functools
import json

from sdc.phase46 import common as cm
from sdc.phase46.ensemble import make
from sdc.phase46.teacher import TRUGGY, LoopExpert
from sdc.phase47 import drills as dl
from sdc.phase47.train_residual import BASE, env_kw

CHECK_SEEDS = list(range(47000, 47034))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    env = env_kw(val=True)
    od = env_kw and None
    res, ref = {}, None
    for car in ["teacher"] + [c for c in a.cars if c != "teacher"]:
        if car == "teacher":
            mk, kw = functools.partial(LoopExpert, **TRUGGY), {k: v for k, v in env.items() if k != "expo"}
        elif car == "base":
            mk, kw = functools.partial(make, tuple(BASE)), env
        else:
            from sdc.jumps import env as je
            od = je.JumpBatch(1, **env).obs_dim + 2
            mk, kw = functools.partial(dl.load_residual, tuple(BASE), car, od), env
        eps = cm.prun(mk, CHECK_SEEDS, procs=a.procs, **kw)
        if car == "teacher":
            ref = eps
        sc = cm.lap_score(eps, teacher=ref)
        res[car] = dict(finished=sc["finished_n"], in_time=sc["in_time_n"], runs=sc["runs"], lap_s=sc["lap_s"],
                        landed=sc["rate"], reasons=sc["reasons"], lap_steps=[e["l"] if e["finished"] else None for e in eps])
        print(car, {k: v for k, v in res[car].items() if k != "lap_steps"}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
