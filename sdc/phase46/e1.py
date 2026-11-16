"""Phase 4.6 E1: what breaks on turns?

    python -m sdc.phase46.e1 --run runs/p45-e5-bc-s0 --out runs/p46-e1

1. Phase 4.5's car (the owner's GRU + filter, copied from the steady expert on STRAIGHT roller lanes; E5's copy,
   rebuilt with the logged command) and the new loop teacher (reads the true state) each drive the validation
   loops: 20 loops x 3 starts = 60 laps, at nominal and at gate-range physics, one car at a time.
2. How early do the parts-list range sensors register a turn? The car is placed (standing, level on its springs)
   on the centreline every 0.25 m before each turn, and its 67 range readings (forward lidar, 8x8 ToF, two side
   sensors; no noise) are compared with the same car on the same stretch laid out straight (same bumps, no turn). "Seen" = any reading differs by
   more than 0.25 m. Compared with where the teacher has to start slowing down for that turn.
Writes <out>/e1.json.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import numpy as np

import mujoco

from sdc.jumps import env as je
from sdc.jumps import loop as lp
from sdc.jumps import real as rl
from sdc.phase46 import common as cm
from sdc.phase46.teacher import LoopExpert


def load_car(run):
    from sdc.phase4 import train_rnn as trr
    pol, _ = trr.load(Path(run))
    return pol


def corners(course, min_peak=1 / 10.0):
    """Turns worth the name: (start s, end s, tightest radius m), in driving order."""
    out = []
    for a, k in lp._runs(np.abs(course.kappa) >= lp.STRAIGHT):
        idx = (a + np.arange(k)) % len(course.kappa)
        peak = np.abs(course.kappa[idx]).max()
        if peak >= min_peak and a + k <= len(course.kappa):
            out.append((a * course.ds, (a + k) * course.ds, 1 / peak))
    return sorted(out)


def _readings(car, s):
    """Noise-free range readings of the car standing on the centreline at s: lidar (1), ToF (64), sides (2)."""
    X, Y, th = car.course.pose(s)
    car.d.qpos[:2] = X, Y
    car.d.qpos[2] = car.course.height(s) + car._dz
    car.d.qpos[3:7] = np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)
    mujoco.mj_forward(car.m, car.d)
    tf_dir = np.array([[np.cos(np.deg2rad(3)), 0, -np.sin(np.deg2rad(3))]])
    return np.concatenate([car._cast((0.235, 0, 0.02), tf_dir, rl.TFMINI_MAX),
                           car._cast((0.235, 0, 0.02), car.tof_dirs, 3.0),
                           car._cast((0, 0, 0.02), np.array([[0, 1.0, 0], [0, -1.0, 0]]), rl.SIDE_MAX)])


def _straightened(c, s_from, s_to):
    """The stretch of loop c from s_from to s_to laid out straight: same bumps, no turn, no banking."""
    idx = np.arange(int(s_from / c.ds), int(s_to / c.ds)) % len(c.center)
    n = len(idx)
    center = np.stack([(np.arange(n) - n / 2) * c.ds, np.zeros(n)], 1)
    return lp.LoopCourse(c.seed, center, np.zeros(n), np.zeros(n), n * c.ds, c.profile[idx], c.rough[idx], np.zeros(n))


def sensor_look(courses, threshold=0.25, back=8.0):
    teacher = LoopExpert()
    car = je.JumpCar(np.random.default_rng(0), real="all", sensors="parts", loop=True)
    rows = []
    for seed in courses:
        car.reset(seed)
        c = car.course
        car._dz = float(car.d.qpos[2] - c.height(lp.START_X))
        plan = teacher.corner_speed(c)
        for s0, s1, radius in corners(c):
            lo = s0 - back
            busy = any(j.start - 1 < s0 and j.end + 4 > lo for j in c.jumps) or (c.whoops[1] > lo and c.whoops[0] < s0)
            prev_turn = np.abs(c.kappa[int((lo - 3.0) / c.ds):int(s0 / c.ds) - 1]).max() >= lp.STRAIGHT if lo > 3.0 else True
            if lo < 3.0 or busy or prev_turn or s0 + 9.0 > c.length:
                continue             # a jump, the whoops, another turn or the line just before this turn: skip it
            twin = _straightened(c, lo - 2.0, s0 + 9.0)
            seen = {"lidar": 0.0, "tof": 0.0, "side": 0.0}
            for dist in np.arange(back, -0.01, -0.25):
                here = _readings(car, s0 - dist)
                car.course = twin
                base = _readings(car, back + 2.0 - dist)
                car.course = c
                diff = np.abs(here - base)
                for name, sl in (("lidar", slice(0, 1)), ("tof", slice(1, 65)), ("side", slice(65, 67))):
                    if seen[name] == 0.0 and diff[sl].max() > threshold:
                        seen[name] = float(dist)
            # where the teacher starts slowing for this turn: walk back from the turn's slowest point while its
            # speed plan is under the cruise speed (metres before the turn's start; negative = inside the turn)
            i0, i1 = int(s0 / c.ds), int(s1 / c.ds)
            k = i0 + int(np.argmin(plan[i0:i1]))
            while k > i0 - int(back / c.ds) and plan[k - 1] < teacher.v_cruise:
                k -= 1
            brake = (i0 - k) * c.ds
            rows.append(dict(course=seed, s=round(s0, 2), radius=round(radius, 2), **seen,
                             any=max(seen.values()), brake=round(float(brake), 2),
                             v_corner=round(float(plan[int((s0 + s1) / 2 / c.ds)]), 2)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/p45-e5-bc-s0")
    ap.add_argument("--out", default="runs/p46-e1")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=cm.MAX_STEPS)
    a = ap.parse_args()
    env = dict(cm.CAR, max_steps=a.max_steps)
    out = dict(scores={}, episodes={}, courses={s: lp.make(s).to_dict() for s in cm.VAL_SEEDS})
    for c in cm.VAL_SEEDS:
        out["courses"][c]["corners"] = corners(lp.make(c))
    for d in ("nominal", "gate"):
        t = cm.prun(LoopExpert, cm.VAL_SEEDS, procs=a.procs, dr=d, **env)
        p = cm.prun(functools.partial(load_car, a.run), cm.VAL_SEEDS, procs=a.procs, dr=d, **env)
        for who, eps in (("teacher", t), ("Phase 4.5 car", p)):
            sc = cm.lap_score(eps, teacher=t)
            out["scores"][f"{who}|{d}"] = sc
            out["episodes"][f"{who}|{d}"] = eps
            print(f"{who:14s} {d:8s} finished {sc['finished_n']}/{sc['runs']}, in time {sc['in_time_n']}, "
                  f"landed {sc['rate']:.0%}, endings {sc['reasons']}", flush=True)
    out["sensors"] = sensor_look(cm.VAL_SEEDS)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, "e1.json").write_text(json.dumps(out, default=float))
    r = out["sensors"]
    print(f"turns measured: {len(r)}; first seen (median): any sensor {np.median([x['any'] for x in r]):.2f} m before "
          f"the turn; teacher starts slowing {np.median([x['brake'] for x in r]):.2f} m before")


if __name__ == "__main__":
    main()
