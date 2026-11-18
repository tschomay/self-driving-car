"""Phase 4.6 gate: python -m sdc.phase46.gate -> runs/p46-gate/gate.json

The gate car: the average of E5's three students (E7). Test loops 46000-46019 x 3 starts, gate-range physics,
physics seed 46 (never used before). For each run:
  out-lap   the remote-driven stand-in's slow lap (teacher.OutLapDriver), as in E3's maps
  timed lap the gate car, one lap from a standing start
  reference the standard true-state teacher's lap on the same run (its time x 1.5 is the limit)
Reports finished laps both with the out-lap counted (a failed out-lap fails the run) and timed lap only.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import torch

from sdc.jumps import driver as dr
from sdc.phase46 import common as cm
from sdc.phase46 import maps
from sdc.phase46.bc_loop import student_env
from sdc.phase46.ensemble import make
from sdc.phase46.teacher import LoopExpert

RUNS = ("runs/p46-e5-s0", "runs/p46-e5-s1", "runs/p46-e5-s2")
OUT = Path("runs/p46-gate")


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    seed = cm.PHYS_SEED_TEST
    ref = cm.prun(LoopExpert, cm.TEST_SEEDS, procs=4, **dict(cm.CAR, dr="gate", phys_seed=seed))
    env = dict(student_env(False, val=True), phys_seed=seed)
    eps = cm.prun(functools.partial(make, RUNS), cm.TEST_SEEDS, procs=4, **env)
    out = maps.collect(cm.TEST_SEEDS, dr.STARTS, "gate", phys_seed=seed)
    keys = [f"{c}|{x0:.2f}" for c in cm.TEST_SEEDS for x0 in dr.STARTS]
    out_ok = [out[k]["reason"] == "finished" for k in keys]
    sc = cm.lap_score(eps, teacher=ref)
    med = sorted(t["l"] for t in ref if t["finished"])[len([t for t in ref if t["finished"]]) // 2]
    in_time = [e["finished"] and e["l"] <= cm.TIME_FACTOR * (t["l"] if t["finished"] else med) for e, t in zip(eps, ref)]
    res = dict(car="average of E5 seeds 0-2", runs=list(RUNS), phys_seed=seed, env=env,
               timed_lap_only=sc["in_time_n"], with_out_lap=int(sum(a and b for a, b in zip(in_time, out_ok))),
               out_laps_failed=int(sum(not x for x in out_ok)),
               out_lap_reasons={k: out[k]["reason"] for k in keys if out[k]["reason"] != "finished"},
               car_score=sc, teacher=cm.lap_score(ref), episodes=eps, teacher_episodes=ref)
    (OUT / "gate.json").write_text(json.dumps(res, indent=1, default=float))
    print("done")


if __name__ == "__main__":
    main()
