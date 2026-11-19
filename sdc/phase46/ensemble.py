"""Phase 4.6 E6: average the actions of several trained students (same inputs), each with its own GRU memory.

python -m sdc.phase46.ensemble --runs runs/p46-e4-nomap-s0 runs/p46-e4-nomap-s1 --out runs/p46-e6-nomap.json [--map]
Scored like bc_loop's final: validation loops x 3 starts, gate-range physics, timed against the true-state teacher.
"""

from __future__ import annotations

import argparse
import functools
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from sdc.phase46 import common as cm
from sdc.phase46.bc_loop import REF, REF_TRUGGY, student_env


class Ensemble:
    def __init__(self, runs):
        self.members = [cm.load_student(r) for r in runs]

    def reset(self, n):
        for m in self.members:
            m.reset(n)

    def __call__(self, b):
        return np.mean([m(b) for m in self.members], axis=0)


def make(runs):
    torch.set_num_threads(1)
    return Ensemble(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--procs", type=int, default=1)
    a = ap.parse_args()
    m0 = json.loads((Path(a.runs[0]) / "metrics.json").read_text())
    env = m0.get("env46") or student_env(a.map, val=True)          # the students' own scoring setup
    truggy = env.get("real") == "truggy"
    eps = cm.prun(functools.partial(make, tuple(a.runs)), cm.VAL_SEEDS, procs=a.procs, **env)
    sc = cm.lap_score(eps, teacher=pickle.load(open(REF_TRUGGY if truggy else REF, "rb")))
    Path(a.out).write_text(json.dumps(dict(runs=a.runs, map=a.map, final=sc, final_episodes=eps), indent=1,
                                      default=float))
    print(f"final: finished {sc['finished_n']}/60, in time {sc['in_time_n']}/60, landed {sc['rate']:.0%}, "
          f"endings {sc['reasons']}")


if __name__ == "__main__":
    main()
