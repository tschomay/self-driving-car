"""Phase 4.6 E9: stitched jump speeds per kind, on the same 200 practice loops as e9_sweep, then the chosen teacher on
validation (60 runs, gate range). python -m sdc.phase46.e9_stitch -> runs/p46-e9-stitch.json"""

from __future__ import annotations

import collections
import functools
import json
import pickle

from sdc.phase46 import common as cm
from sdc.phase46.e9_sweep import LOOPS, jump_stats
from sdc.phase46.teacher import LoopExpert

COMBOS = ({"drop": 6.0}, {"drop": 6.7}, {"drop": 6.0, "double": 7.9}, {"drop": 6.0, "double": 7.0})


def teacher(jv):
    return functools.partial(LoopExpert, a_lat=2.5, a_brake=2.5, v_target=7.4, v_cruise=7.4, jump_v=jv)


def main():
    res = []
    for jv in COMBOS:
        eps = cm.prun(teacher(jv), LOOPS, starts=(0.0,), procs=4, **dict(cm.CAR, dr="gate"))
        r = dict(jump_v=jv, finished=sum(e["finished"] for e in eps),
                 reasons=dict(collections.Counter(e["reason"] for e in eps)), jumps=jump_stats(eps),
                 lap_s=cm.lap_score(eps)["lap_s"])
        res.append(r)
        print(r, flush=True)
    best = max(res, key=lambda r: r["finished"])
    ref = pickle.load(open("runs/p46-teacher-val-gate.pkl", "rb"))
    val = {}
    for name, jv in (("cautious", {}), ("stitched", best["jump_v"])):
        eps = cm.prun(teacher(jv), cm.VAL_SEEDS, procs=4, **dict(cm.CAR, dr="gate"))
        sc = cm.lap_score(eps, teacher=ref)
        val[name] = dict(jump_v=jv, finished=sc["finished_n"], in_time=sc["in_time_n"], reasons=sc["reasons"],
                         lap_s=sc["lap_s"])
        print(name, val[name], flush=True)
    json.dump(dict(practice=res, best=best, validation=val), open("runs/p46-e9-stitch.json", "w"), indent=1)


if __name__ == "__main__":
    main()
