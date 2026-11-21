"""Phase 4.7: where does each driver lose finishes, by the run's physics? Selection runs, gate range, 1 noisy pass.

    python -m sdc.phase47.by_grip CKPT [CKPT ...]   -> runs/p47-by-grip.json
Physics are drawn from (course, start), so they are rebuilt here by resetting a car the same way.
"""
import functools
import json
import sys
from pathlib import Path

import numpy as np

from sdc.jumps import driver as dr
from sdc.jumps import env as je
from sdc.phase46 import common as cm
from sdc.phase47 import e2_prep, racer

OUT = Path("runs/p47-by-grip.json")


def physics(courses, env):
    out = []
    for c in courses:
        for x0 in dr.STARTS:
            b = je.JumpBatch(1, courses=[c], **env)
            b.cars[0].reset(c, x0=x0)
            p = b.cars[0].phys
            out.append(dict(friction=p.friction, motor=p.motor, mass=p.mass))
    return out


def main():
    courses = json.loads(e2_prep.OUT.read_text())["split"]["select"]
    env = racer.env_eval()
    ph = physics(courses, env)
    res = {}
    for name in ["teacher"] + sys.argv[1:]:
        make = (functools.partial(racer._make_teacher, 0) if name == "teacher"
                else functools.partial(racer._make_driver, name, 0))
        eps = cm.prun(make, courses, procs=4, **env)
        res[name] = [dict(finished=bool(e["finished"]), reason=e["reason"], **p) for e, p in zip(eps, ph)]
        f = np.array([r["friction"] for r in res[name]])
        ok = np.array([r["finished"] for r in res[name]])
        rows = {f"{lo}-{hi}": f"{ok[(f >= lo) & (f < hi)].sum()}/{((f >= lo) & (f < hi)).sum()}"
                for lo, hi in ((0.5, 0.7), (0.7, 0.9), (0.9, 1.1), (1.1, 1.3))}
        print(name, "finished by friction:", rows, flush=True)
    OUT.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
