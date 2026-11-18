"""Phase 4.6 E9: the cautious teacher's jump approach speed, per jump kind (owner: "try a few settings and stitch
the best ones together"). python -m sdc.phase46.e9_sweep -> runs/p46-e9-sweep.json

Drives the cautious teacher (a_lat 2.5, a_brake 2.5) on 200 practice loops at gate-range physics with one jump
approach speed for every jump, for each speed in SPEEDS, and scores each jump kind by how often a jump of that kind
was passed cleanly (landed, and the run didn't end there). Practice loops only: validation checks the stitched result.
"""

from __future__ import annotations

import collections
import functools
import json

from sdc.phase46 import common as cm
from sdc.phase46.teacher import LoopExpert

SPEEDS = (6.0, 7.4, 8.5, 9.5)
LOOPS = cm.PRACTICE[:200]


def jump_stats(eps):
    out = collections.defaultdict(lambda: [0, 0])
    for e in eps:
        for j in e["jumps"]:
            ok = j["landed"] and (e["finished"] or e["x"] > j["touch_x"] + 15)
            out[j["kind"]][0] += ok
            out[j["kind"]][1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def main():
    res = {}
    for v in SPEEDS:
        eps = cm.prun(functools.partial(LoopExpert, a_lat=2.5, a_brake=2.5, v_target=v, v_cruise=7.4), LOOPS,
                      starts=(0.0,), procs=4, **dict(cm.CAR, dr="gate"))
        res[v] = dict(finished=sum(e["finished"] for e in eps), reasons=dict(collections.Counter(e["reason"] for e in eps)),
                      jumps=jump_stats(eps), lap_s=cm.lap_score(eps)["lap_s"])
        print(v, res[v], flush=True)
    json.dump(res, open("runs/p46-e9-sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
