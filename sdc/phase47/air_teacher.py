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

GAINS = [(6.0, 1.0), (3.0, 0.5)]
# full-size truggy (size8): the double's take-off speed (racer.TEACHER: 7.4 m/s, tuned on the small car)
DOUBLE_V = [7.0, 7.8, 8.2]


def _make(kp, kd, seed, double_v=None):
    over = dict(kp=kp, kd=kd)
    if double_v:
        over["jump_v"] = {"double": double_v}
    return Noisy(LoopExpert(**dict(racer.TEACHER, **over)), seed * 1000 + os.getpid())


def main():
    courses = json.loads(e2_prep.OUT.read_text())["split"]["select"]
    import sys
    out = {}
    runs = [(6.0, 1.0, v) for v in DOUBLE_V] if "--double" in sys.argv else [(kp, kd, None) for kp, kd in GAINS]
    for kp, kd, dv in runs:
        fin, laps, why = [], [], {}
        for r in range(2):
            eps = cm.prun(functools.partial(_make, kp, kd, r, dv), courses, procs=4, **racer.env_eval())
            fin.append(int(sum(bool(e["finished"]) for e in eps)))
            laps += [e["l"] * 0.02 for e in eps if e["finished"]]
            for e in eps:
                if not e["finished"]:
                    why[e["reason"]] = why.get(e["reason"], 0) + 1
        lap = float(np.median(laps))
        res = dict(kp=kp, kd=kd, double_v=dv, finished_per_pass=fin, lap_s=round(lap, 2),
                   race_s=round(20 * lap + 20 * (1 - np.mean(fin) / 150) * 7, 1), endings=why)
        out[f"{kp}/{kd}/{dv}"] = res
        print(res, flush=True)
    real = racer.env_eval()["real"]
    (__import__("pathlib").Path(f"runs/p47-air-teacher-{real}{'-double' if '--double' in sys.argv else ''}.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
