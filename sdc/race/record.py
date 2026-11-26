"""Record a 20-lap race on a loop, frame by frame, for the replay viewer (issue #86).

    python -m sdc.race.record --car teacher --loops 10001 10002 10003 --out runs/race
    python -m sdc.race.record --car runs/p47-e5-s0 runs/p47-e5-s1 --name e5 ...

One long episode per loop: the finish line is moved out to LAPS laps (the env's track position keeps counting past
the line). The car is the Phase 4.7 setup: truggy physics, steering expo 0.4 for the students (the teacher inverts
expo itself), sensors-only, gate-range physics. One physics draw per loop (phys_seed), kept for the whole race.

Race rule (Phase 4.7, owner): total time for LAPS laps, each crash costs CRASH_S seconds. A crash is any ending the
env reports (rolled over, off the lane, stuck), or spinning round: facing more than SPUN_DEG from the track's direction
for SPUN_S (the env never ends a lap for that, and a spun car can drive the loop backwards for minutes). The recording carries on for TUMBLE_S with no throttle so the crash
is visible, holds the car still for the rest of the penalty (the marshal's countdown), then the car is put back on
the centreline at the furthest point it reached (or RUNUP_M before a jump's kicker if it crashed on a jump), standing, with a fresh
memory (as if switched off and on). Race time = sim time + CRASH_S per crash; the frames are on race time.

Output per (car, loop): <out>/<name>-<loop>.json.gz with the frames (FPS Hz, fixed-point integer columns) and
events, and once per loop <out>/track-<loop>.json.gz with the terrain and centreline for the viewer.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch

from sdc.jumps import car as cr
from sdc.jumps import course as crs
from sdc.jumps import env as je
from sdc.jumps import loop as lp

LAPS = 20
CRASH_S = 7.0
TUMBLE_S = 2.0
RUNUP_M = 9.0
SPUN_DEG, SPUN_S = 120.0, 0.5
MAX_LAP_S = 120.0                        # a lap this long ends the race (DNF): a safety net
FPS = 25
EVERY = int(round(1 / (FPS * cr.DT)))    # control steps per frame (2)
LOOPS = (10001, 10002, 10003)            # validation loops

# frame columns and their fixed-point scales (value = int / scale)
COLS = dict(x=1000, y=1000, z=1000, qw=10000, qx=10000, qy=10000, qz=10000,
            s1=1000, s2=1000, s3=1000, s4=1000,           # suspension travel per wheel (fl fr rl rr), m
            st=1000,                                      # front wheel steer angle (mean of the two), rad
            spin=100,                                     # mean wheel spin angle, rad (wrapped to +-pi)
            v=100,                                        # forward speed, m/s
            cs=1000, ct=1000,                             # steering and throttle commands
            air=1, gnd=1)                                 # all wheels off the ground; wheels down (bit mask)

# the property is looked up on the class: give it an instance override for the long race
lp.LoopCourse.finish = property(lambda self: getattr(self, "_finish", self.length))


def driver(car):
    """(policy, env kwargs, label). car: 'teacher', 'base', 'e4', or run directories (averaged)."""
    from sdc.phase46.ensemble import Ensemble
    from sdc.phase47 import where
    if car in (["teacher"], ["base"], ["e4"]):
        pol, env = where._driver(car[0])
        return pol, env, car[0]
    pol, env = Ensemble(tuple(car)), where.env_kw(val=True)
    return pol, env, Path(car[0]).name.rsplit("-s", 1)[0]


def _reset_at(c, seed, total, phys):
    """Put the car back standing at track position `total` (counting laps), with the race's physics."""
    x0 = (total % c.course.length) - crs.START_X
    c.reset(seed, x0=x0)
    if not c.legacy:
        c.phys = phys
        c._apply_phys()
    c.course._finish = LAPS * c.course.length
    c._s = total
    c.x_prev = total


def race(car, seed, phys_seed=0, laps=LAPS, log=print):
    pol, env, label = driver(car)
    torch.set_num_threads(1)
    env = dict(env, phys_seed=phys_seed)
    b = je.JumpBatch(1, courses=[seed], **{**env, "max_steps": 10**9})
    c = b.cars[0]
    b._obs = np.zeros((1, b.obs_dim), np.float32)
    b._obs[0] = c.reset(seed)
    course = c.course
    L = course.length
    course._finish = laps * L
    phys = c.phys
    if hasattr(pol, "reset"):
        pol.reset(1)
    m, d = c.m, c.d
    qa = lambda j: m.jnt_qposadr[m.joint(j).id]     # noqa: E731
    susp = np.array([qa(f"{w}_susp") for w in cr.WHEELS])
    steer = np.array([qa(f"{w}_steer") for w in ("fl", "fr")])
    spin = np.array([qa(f"{w}_spin") for w in cr.WHEELS])

    frames, events = [], []
    t_race = 0.0           # race clock, s
    lap_marks = [0.0]      # race time at each line crossing (the first lap starts at the gun)
    was_air, takeoff = False, None
    crashes = 0
    far = c.state()["x"]   # furthest progress so far (a spun car can go backwards)
    wrong = 0
    dnf = False

    def frame(a, v, air, wheels):
        q = d.xquat[1]
        sp = float(np.mean(d.qpos[spin]))
        frames.append((*d.xpos[1], *q, *d.qpos[susp], float(np.mean(d.qpos[steer])),
                       (sp + np.pi) % (2 * np.pi) - np.pi, v, float(a[0]), float(a[1]), float(air),
                       float(sum(int(w) << k for k, w in enumerate(wheels)))))

    t0 = time.time()
    while True:
        a = np.asarray(pol(b), np.float32)[0]
        _, term, trunc, reason = c.step(a, max_steps=10**9)
        b._obs[0] = c._obs
        t_race += cr.DT
        st = c.state()
        wheels, _ = c.contacts()
        # airtime (events for the HUD and the slow-motion replays)
        if c.airborne and not was_air:
            takeoff = (t_race, st["x"])
        if was_air and not c.airborne and takeoff is not None:
            air = t_race - takeoff[0]
            if air >= je.MIN_AIR:
                events.append(dict(t=round(takeoff[0], 3), type="jump", air=round(air, 3),
                                   s=round(takeoff[1] % L, 2), jump=course.jump_at(takeoff[1]),
                                   lap=len(lap_marks)))
        was_air = c.airborne
        far = max(far, st["x"])
        wrong = wrong + 1 if abs(st["yaw"]) > np.deg2rad(SPUN_DEG) else 0
        if wrong >= SPUN_S / cr.DT and not (term or trunc):
            term, reason = True, "spun round"
        # line crossings
        while st["x"] >= len(lap_marks) * L and len(lap_marks) <= laps:
            lap_marks.append(round(t_race, 3))
            events.append(dict(t=round(t_race, 3), type="lap", lap=len(lap_marks) - 1,
                               time=round(lap_marks[-1] - lap_marks[-2], 3)))
            log(f"  {label} loop {seed}: lap {len(lap_marks) - 1} {lap_marks[-1] - lap_marks[-2]:.2f} s "
                f"(race {t_race:.1f} s, crashes {crashes}, wall {time.time() - t0:.0f} s)")
        if c.t % EVERY == 0:
            frame(a, st["v_fwd"], c.airborne, wheels)
        if reason == "finished" or len(lap_marks) > laps:
            break
        if t_race - lap_marks[-1] > MAX_LAP_S:
            dnf = True
            log(f"  {label} loop {seed}: DNF, lap {len(lap_marks)} over {MAX_LAP_S} s")
            break
        if term or trunc:
            crashes += 1
            wrong = 0
            total = far
            events.append(dict(t=round(t_race, 3), type="crash", reason=reason, s=round(total % L, 2),
                               lap=len(lap_marks), until=round(t_race + CRASH_S, 3)))
            # the crash plays out (no throttle, wheels straight), then the car sits until the marshal gets there
            n_tumble = int(TUMBLE_S / cr.DT)
            for k in range(int(CRASH_S / cr.DT)):
                if k < n_tumble:
                    c._physics(np.zeros(2, np.float32))
                c.t += 1
                t_race += cr.DT
                if c.t % EVERY == 0:
                    frame((0.0, 0.0), float(c.state()["v_fwd"]) if k < n_tumble else 0.0, False, c.contacts()[0])
            # back on the centreline where it crashed; on a jump, RUNUP_M before the kicker so it can make it
            k = course.jump_at(total)
            back = total
            if k is not None:
                j = course.jumps[k]
                lap0 = total - (total % L)
                back = lap0 + j.start - RUNUP_M
            back = max(back, total - 25.0)
            far = back
            _reset_at(c, seed, back, phys)
            if hasattr(pol, "reset"):
                pol.reset(1)
            b._obs[0] = c._obs
            events.append(dict(t=round(t_race, 3), type="reset", s=round(back % L, 2)))
            log(f"  {label} loop {seed}: crash {crashes} ({reason}) at {total % L:.1f} m, lap {len(lap_marks)}")
    lap_times = [round(b2 - a2, 3) for a2, b2 in zip(lap_marks, lap_marks[1:])]
    fr = np.array(frames, np.float64)
    cols = {}
    for i, (k, sc) in enumerate(COLS.items()):
        cols[k] = np.rint(fr[:, i] * sc).astype(np.int32).tolist()
    geom = (cr.geometry(c.real) if hasattr(cr, "geometry")         # the full-size truggy ("size8") and later sims
            else dict(wheel_r=cr.WHEEL_R, corner_x=0.15, corner_y=0.14, mount_z=cr.MOUNT_Z, box=(0.23, 0.09, 0.035)))
    geom = {k: geom[k] for k in ("wheel_r", "corner_x", "corner_y", "mount_z", "box")}
    return dict(car=label, loop=seed, phys_seed=phys_seed, geom=geom, real=sorted(c.real), phys={k: round(v, 4) for k, v in vars(phys).items()},
                laps=laps, crash_s=CRASH_S, fps=FPS, n=len(frames), race_time=round(t_race, 3),
                lap_times=lap_times, best_lap=min(lap_times) if lap_times else None, crashes=crashes, dnf=dnf,
                scales=COLS, cols=cols, events=events)


def track(seed):
    """The loop for the viewer: centreline, jumps, and the terrain (cropped to the track, every other cell)."""
    c = lp.make(seed)
    hf = c.hfield() * lp.ZMAX
    step = 2
    pad = lp.STRIP + 4.0
    lo, hi = c.center.min(0) - pad, c.center.max(0) + pad
    ix0, iy0 = [int(np.floor((v + lp.BOX / 2) / lp.D / step)) * step for v in lo]
    ix1, iy1 = [min(int(np.ceil((v + lp.BOX / 2) / lp.D / step)) * step, lp.N - 1) for v in hi]
    ix0, iy0 = max(ix0, 0), max(iy0, 0)
    sub = hf[iy0:iy1 + 1:step, ix0:ix1 + 1:step]
    ny, nx = sub.shape
    stride = 4   # centreline every 0.2 m
    return dict(seed=seed, length=round(c.length, 3), ds=lp.DS * stride, drive_w=lp.DRIVE_W, bank_w=lp.BANK_W,
                bank_h=lp.BANK_H, base=lp.BASE, start_x=crs.START_X,
                center=np.round(c.center[::stride], 3).tolist(), heading=np.round(c.heading[::stride], 4).tolist(),
                kappa=np.round(c.kappa[::stride], 4).tolist(),
                profile=np.round(c.profile[::stride], 4).tolist(), bank=np.round(c.bank[::stride], 4).tolist(),
                jumps=[j.to_dict() for j in c.jumps], whoops=[round(w, 2) for w in c.whoops],
                hf=dict(nx=nx, ny=ny, x0=round(-lp.BOX / 2 + ix0 * lp.D, 3), y0=round(-lp.BOX / 2 + iy0 * lp.D, 3),
                        cell=lp.D * step, mm=np.rint(sub * 1000).astype(np.int32).ravel().tolist()))


def _save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(obj, f, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--car", nargs="+", default=["teacher"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--loops", nargs="+", type=int, default=list(LOOPS))
    ap.add_argument("--laps", type=int, default=LAPS)
    ap.add_argument("--phys-seed", type=int, default=0)
    ap.add_argument("--out", default="runs/race")
    a = ap.parse_args()
    out = Path(a.out)
    for seed in a.loops:
        tp = out / f"track-{seed}.json.gz"
        if not tp.exists():
            _save(track(seed), tp)
        r = race(a.car, seed, a.phys_seed, a.laps)
        name = a.name or r["car"]
        r["car"] = name
        _save(r, out / f"{name}-{seed}.json.gz")
        print(f"{name} loop {seed}: race {r['race_time']:.1f} s, best lap {r['best_lap']} s, crashes {r['crashes']}, "
              f"{r['n']} frames")


if __name__ == "__main__":
    main()
