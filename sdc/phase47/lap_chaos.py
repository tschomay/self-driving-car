"""Phase 4.7, after E2: are full-lap finishes coin flips too?

    python -m sdc.phase47.lap_chaos      -> runs/p47-lap-chaos.json

The base car and the teacher each drive E2's 150 selection runs (50 held-out practice loops x 3 starts, gate range)
REPEATS times with tiny independent action noise (std NOISE, about 1% of full range). Per run: how often it finished.
A run that finishes on some repeats and not others is decided by chance.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

from sdc.phase46 import common as cm
from sdc.phase46.ensemble import Ensemble
from sdc.phase46.teacher import TRUGGY, LoopExpert
from sdc.phase47 import e2_prep
from sdc.phase47.train_residual import BASE, env_kw

OUT = Path("runs/p47-lap-chaos.json")
NOISE, REPEATS = 0.02, 3


class Noisy:
    def __init__(self, inner, seed):
        self.inner, self.rng = inner, np.random.default_rng(seed)
        self.env_kw = getattr(inner, "env_kw", {})

    def reset(self, n):
        if hasattr(self.inner, "reset"):
            self.inner.reset(n)

    def __call__(self, b):
        a = np.asarray(self.inner(b), np.float32)
        return np.clip(a + self.rng.normal(0, NOISE, a.shape), -1, 1).astype(np.float32)


def make(who, seed):
    import os
    s = seed * 1000 + os.getpid()          # each worker its own noise stream
    return Noisy(Ensemble(BASE) if who == "base" else LoopExpert(**TRUGGY), s)


def main():
    courses = json.loads(e2_prep.OUT.read_text())["split"]["select"]
    env = env_kw(val=True)
    out = {}
    for who in ("base", "teacher"):
        kw = env if who == "base" else {k: v for k, v in env.items() if k != "expo"}
        fin = []
        for r in range(REPEATS):
            eps = cm.prun(functools.partial(make, who, r), courses, procs=4, **kw)
            fin.append([bool(e["finished"]) for e in eps])
            print(who, r, sum(fin[-1]), flush=True)
        f = np.array(fin).mean(0)
        out[who] = dict(finished_per_repeat=[int(sum(x)) for x in fin], always=int((f == 1).sum()),
                        never=int((f == 0).sum()), mixed=int(((f > 0) & (f < 1)).sum()), finished=fin)
        print(who, {k: v for k, v in out[who].items() if k != "finished"}, flush=True)
    OUT.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
