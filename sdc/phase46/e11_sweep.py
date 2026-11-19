"""Phase 4.6 E11 prep: the truggy teacher's jump speed per jump kind (E9's stitching, on the truggy physics).

    python -m sdc.phase46.e11_sweep      # -> runs/p46-e11-sweep.json

One jump speed for every jump, for each speed in SPEEDS, on 100 practice loops at gate range (expo 0.4, as the
students drive); per kind, how often a jump was passed cleanly. Then the stitched speeds on the same loops, and the
stitched teacher against the plain one on validation.
"""

from __future__ import annotations

import collections
import functools
import json
import pickle

from sdc.phase46 import common as cm
from sdc.phase46.e9_sweep import jump_stats
from sdc.phase46.teacher import TRUGGY, LoopExpert

SPEEDS = (6.0, 6.8, 7.6, 8.4)
LOOPS = cm.PRACTICE[:100]
ENV = dict(cm.CAR, real="truggy", dr="gate", expo=0.4)


def run(loops, starts=(0.0,), **kw):
    eps = cm.prun(functools.partial(LoopExpert, **{**TRUGGY, **kw}), loops, starts=starts, procs=4, **ENV)
    return eps, dict(finished=sum(e["finished"] for e in eps), reasons=dict(collections.Counter(e["reason"] for e in eps)),
                     jumps=jump_stats(eps), lap_s=cm.lap_score(eps)["lap_s"])


def main():
    res = {}
    for v in SPEEDS:
        _, res[v] = run(LOOPS, v_target=v)
        print(v, res[v], flush=True)
    kinds = res[SPEEDS[0]]["jumps"].keys()
    best = {k: max(SPEEDS, key=lambda v: res[v]["jumps"][k][0] / max(1, res[v]["jumps"][k][1])) for k in kinds}
    _, st = run(LOOPS, jump_v=best)
    print("stitched", best, st, flush=True)
    ref = pickle.load(open("runs/p46-teacher-val-gate-truggy.pkl", "rb"))
    val = {}
    for name, jv in (("plain", {}), ("stitched", best)):
        eps, v = run(cm.VAL_SEEDS, starts=(0.0, -0.3, 0.3), jump_v=jv)
        v["in_time"] = cm.lap_score(eps, teacher=ref)["in_time_n"]
        val[name] = v
        print(name, v, flush=True)
    json.dump(dict(sweep={str(k): v for k, v in res.items()}, best=best, stitched_practice=st, validation=val),
              open("runs/p46-e11-sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
