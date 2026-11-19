"""Phase 4.7, after E2: can the exam drills be recovered at all?

    python -m sdc.phase47.recoverable      -> runs/p47-recoverable.json

On E2's 300 exam drills (same starts, same perturbations):
  teacher   the hand-coded LoopExpert (truggy settings), reading the true state; steering passed through the inverse
            of the 0.4 expo so its steering angles are its own.
  grid      the base car plus one fixed correction at full residual strength (steer and throttle each -0.3, 0 or +0.3:
            9 options) held for the first HOLD steps, then nothing. A drill counts as recoverable if any option avoids
            the crash: an existence check for what the residual could learn, not a driver.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from sdc.jumps.env import expo_inverse
from sdc.phase46.teacher import TRUGGY, LoopExpert
from sdc.phase47 import drills2 as d2
from sdc.phase47 import e2_prep
from sdc.phase47.drills import CRASHES, DRILL_STEPS
from sdc.phase47.train_residual import BASE, env_kw

OUT = Path("runs/p47-recoverable.json")
HOLD = 50          # 1 s
GRID = [(s, t) for s in (-1, 0, 1) for t in (-1, 0, 1)]
_EXAM = None


def teacher_run(starts):
    """Each start driven by the teacher alone. Returns crashed flags."""
    kw = env_kw()
    runner = d2.Runner(BASE, 1, kw)
    t = LoopExpert(**TRUGGY)
    out = []
    for s in starts:
        runner.start(0, s)
        car = runner.cars[0]
        crashed = False
        while True:
            a = t.act(car)
            a[0] = expo_inverse(float(a[0]), car.expo)
            r, done, tc, reason = car.step(a, 0.0, int(runner.end_t[0]), 0.0, 0.0)
            if runner.kick[0] and car.airborne and car.course.jump_at(car.takeoff[1]) is not None:
                car.d.qvel[4] += runner.kick[0]
                runner.kick[0] = 0.0
            if reason in CRASHES:
                crashed = True
                break
            if done or tc:
                break
        out.append(crashed)
    return out


def grid_run(starts, option):
    """Each start driven by the base car + a fixed full-strength correction for HOLD steps."""
    raw_hold = np.array(option, np.float32) * 4.0          # 0.3 * tanh(4) = 0.2998
    n = 8
    runner = d2.Runner(BASE, n, env_kw())
    out = [None] * len(starts)
    queue = list(range(len(starts)))
    active = [None] * n
    t0 = np.zeros(n, int)
    for i in range(n):
        active[i] = queue.pop(0) if queue else None
        runner.start(i, starts[active[i] if active[i] is not None else 0])
        t0[i] = runner.cars[i].t
    while any(a is not None for a in active):
        runner.obs()
        raw = np.array([raw_hold if c.t - t0[i] < HOLD else np.zeros(2, np.float32)
                        for i, c in enumerate(runner.cars)], np.float32)
        _, crashed, ended, _ = runner.step(raw)
        for i in np.flatnonzero(ended | crashed):
            if active[i] is not None:
                out[active[i]] = bool(crashed[i])
            active[i] = queue.pop(0) if queue else None
            runner.start(i, starts[active[i] if active[i] is not None else 0])
            t0[i] = runner.cars[i].t
    return out


def _job(j):
    torch.set_num_threads(1)
    if j == "teacher":
        return j, teacher_run(_EXAM)
    return j, grid_run(_EXAM, GRID[j])


def main():
    global _EXAM
    prep = json.loads(e2_prep.OUT.read_text())
    _EXAM, _ = e2_prep.exam_set(prep["amps"])
    base = prep["base_exam_crashed"]
    with mp.get_context("fork").Pool(4) as pool:
        res = dict(pool.imap_unordered(_job, ["teacher"] + list(range(len(GRID)))))
    grid = np.array([res[j] for j in range(len(GRID))])                # (9, 300)
    any_ok = ~grid.all(0)
    kinds = [s.kind for s in _EXAM]
    by = lambda flags: {k: round(float(np.mean([f for f, kk in zip(flags, kinds) if kk == k])), 3)   # noqa: E731
                        for k in d2.KINDS} | dict(all=round(float(np.mean(flags)), 3))
    out = dict(base=by(base), teacher=by(res["teacher"]),
               grid={str(GRID[j]): by(grid[j]) for j in range(len(GRID))},
               no_option_saves=by(~any_ok), base_crashed_but_some_option_saves=by(np.array(base) & any_ok),
               teacher_crashed=res["teacher"], grid_crashed=grid.tolist())
    OUT.write_text(json.dumps(out, indent=1))
    for k in ("base", "teacher", "no_option_saves", "base_crashed_but_some_option_saves"):
        print(k, out[k])
    for g, v in out["grid"].items():
        print("grid", g, v)


if __name__ == "__main__":
    main()
