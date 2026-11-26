"""Phase 4.7, after the air-control fix: the racer's starting teacher on the truggy2 physics, with its air gains
(kp, kd: pitch control in the air) as they were and scaled down. Selection runs, 2 noisy passes each.

    SDC_REAL=truggy2 python -m sdc.phase47.air_teacher
"""
import functools
import json
import os

import numpy as np

from sdc.phase46 import common as cm
from sdc.phase46.teacher import LoopExpert
from sdc.phase47 import e2_prep, racer
from sdc.phase47.lap_chaos import Noisy

GAINS = [(6.0, 1.0), (4.0, 0.7), (3.0, 0.5), (2.0, 0.35)]


def _make(kp, kd, seed):
    return Noisy(LoopExpert(**dict(racer.TEACHER, kp=kp, kd=kd)), seed * 1000 + os.getpid())


def main():
    courses = json.loads(e2_prep.OUT.read_text())["split"]["select"]
    out = {}
    for kp, kd in GAINS:
        fin, laps, why = [], [], {}
        for r in range(2):
            eps = cm.prun(functools.partial(_make, kp, kd, r), courses, procs=4, **racer.env_eval())
            fin.append(int(sum(bool(e["finished"]) for e in eps)))
            laps += [e["l"] * 0.02 for e in eps if e["finished"]]
            for e in eps:
                if not e["finished"]:
                    why[e["reason"]] = why.get(e["reason"], 0) + 1
        lap = float(np.median(laps))
        res = dict(kp=kp, kd=kd, finished_per_pass=fin, lap_s=round(lap, 2),
                   race_s=round(20 * lap + 20 * (1 - np.mean(fin) / 150) * 7, 1), endings=why)
        out[f"{kp}/{kd}"] = res
        print(res, flush=True)
    real = racer.env_eval()["real"]
    (__import__("pathlib").Path(f"runs/p47-air-teacher-{real}.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
