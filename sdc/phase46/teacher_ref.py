"""The re-tuned teacher's validation laps on the truggy physics: the time reference for students' laps.

    python -m sdc.phase46.teacher_ref     # -> runs/p46-teacher-val-gate-truggy.pkl (gitignored; rebuild in a new container)
"""

from __future__ import annotations

import functools
import pickle

from sdc.phase46 import common as cm
from sdc.phase46.bc_loop import REF_TRUGGY
from sdc.phase46.teacher import TRUGGY, LoopExpert


def main():
    eps = cm.prun(functools.partial(LoopExpert, **TRUGGY), cm.VAL_SEEDS, starts=(0.0, -0.3, 0.3), procs=4,
                  **dict(cm.CAR, real="truggy", dr="gate"))
    REF_TRUGGY.parent.mkdir(exist_ok=True)
    pickle.dump(eps, open(REF_TRUGGY, "wb"))
    print("teacher finished", sum(e["finished"] for e in eps), "/", len(eps))


if __name__ == "__main__":
    main()
