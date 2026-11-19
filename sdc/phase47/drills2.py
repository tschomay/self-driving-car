"""Phase 4.7 E2: harder drills, calibrated per kind, and a fixed drill exam.

Changes from E1's drills (drills.py, kept unchanged for reproducibility):
  - Harvest with slowly varying noise (HARVEST_NOISE, correlation HARVEST_RHO per step) on the base car's actions, so it crashes far more often and leaves
    many crash starts. Every on-the-ground snapshot is kept too, as starts for the edge and slide drills.
  - Five kinds, each with a strength `amp` in [0, 1] (calibrated so the base car alone crashes ~TARGET of the time):
      corner   before a sharp turn; speed x (1 + 1.5 amp u)
      takeoff  a lip 4-9 m ahead; speed x (1 +/- 0.5 amp u) and a pitch-rate kick +/- 5 amp u rad/s off the lip
      edge     anywhere on the ground; a sideways push toward the nearer bank, 7 amp u m/s
      slide    anywhere on the ground; a yaw-rate burst +/- 16 amp u rad/s
      crash    1-3 s before a place the (noisy) base car went off the lane or rolled over; speed x 0.95-1.05
    with u ~ U(0.5, 1). Perturbations are drawn once per drill (Start), so an exam drill is the same every time.
  - run_starts: drives a list of Starts with a greedy residual (or none = the base car) and reports crashes. Used for
    calibration and for the exam.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch

from sdc.jumps import env as je
from sdc.phase4 import train as tr
from sdc.phase46.hard import snapshot
from sdc.phase47.drills import (CRASHES, DRILL_CRASHES, DRILL_STEPS, SNAP_EVERY, Base, Drill, Slot, _features,
                                restore)

KINDS = ("corner", "takeoff", "edge", "slide", "crash")
TUNED = ("corner", "takeoff", "edge", "slide")
HARVEST_NOISE, HARVEST_RHO = 0.35, 0.9
TARGET = 0.4


@dataclass
class Start:
    drill: Drill
    kind: str
    p: dict = field(default_factory=dict)    # speed, kick, push, yaw


def harvest(base_runs, courses, env_kw, seed=0, n=8, max_steps=2500, noise=HARVEST_NOISE, keep_ground=0.25):
    """Drive the noisy base car on `courses` (each exactly once); return {kind: [Drill]} and a summary. "ground"
    holds a random `keep_ground` share of on-the-ground snapshots (edge and slide starts)."""
    rng = np.random.default_rng(seed)
    b = je.JumpBatch(n, seed=seed, courses=list(courses), **{**env_kw, "max_steps": max_steps, "jump_bonus": 0.0})
    base = Base(base_runs, n)
    obs = b.reset()
    pool = {k: [] for k in ("corner", "takeoff", "ground", "crash")}
    ring = [collections.deque(maxlen=7) for _ in range(n)]
    quota = [len(list(courses)[i::n]) for i in range(n)]
    done_n = [0] * n
    endings = collections.Counter()
    eps = np.zeros((n, 2))
    while any(d < q for d, q in zip(done_n, quota)):
        hs = [base.get(i) for i in range(n)]
        eps = HARVEST_RHO * eps + np.sqrt(1 - HARVEST_RHO ** 2) * noise * rng.standard_normal((n, 2))
        a = np.clip(base.act(obs) + eps, -1, 1)
        for i, car in enumerate(b.cars):
            if done_n[i] >= quota[i]:
                continue
            if car.t % SNAP_EVERY == 0 and car.t > 50:
                snap_car, state = snapshot(car)
                d = Drill("?", snap_car, state, hs[i])
                ring[i].append((car.t, d))
                k_now, k_ahead, lip = _features(car)
                if k_now < 0.08 and k_ahead > 0.15:
                    pool["corner"].append(Drill("corner", snap_car, state, hs[i]))
                elif 4.0 <= lip <= 9.0:
                    pool["takeoff"].append(Drill("takeoff", snap_car, state, hs[i]))
                if not car.airborne and car.course.jump_at(car.state()["x"]) is None and rng.random() < keep_ground:
                    pool["ground"].append(Drill("ground", snap_car, state, hs[i]))
        obs, _, term, trunc, infos = b.step(a)
        for i in np.flatnonzero(term | trunc):
            ep = infos[i]["episode"]
            if done_n[i] < quota[i]:
                endings[ep["reason"]] += 1
            done_n[i] += 1
            if done_n[i] <= quota[i] and ep["reason"] in DRILL_CRASHES:
                t_end = ep["l"]
                for t, d in ring[i]:
                    if any(abs((t_end - t) - back) <= SNAP_EVERY / 2 for back in (50, 100, 150)):
                        pool["crash"].append(Drill("crash", d.car, d.state, d.h))
            ring[i].clear()
            base.set(i, [torch.zeros_like(h) for h in hs[i]], started=True)
    for v in pool.values():
        rng.shuffle(v)
    return pool, dict(laps=sum(quota), endings=dict(endings), **{k: len(v) for k, v in pool.items()})


def draw(pool, kind, amp, rng):
    """One Start of `kind` at strength `amp` (amp is ignored for crash drills)."""
    src = pool["ground"] if kind in ("edge", "slide") else pool[kind]
    assert src, f"no harvested starts for {kind} drills"
    d = src[rng.integers(len(src))]
    u = rng.uniform(0.5, 1.0)
    sgn = float(rng.choice([-1.0, 1.0]))
    if kind == "corner":
        p = dict(speed=1 + 1.5 * amp * u)
    elif kind == "takeoff":
        p = dict(speed=1 + sgn * 0.5 * amp * u, kick=float(rng.choice([-1.0, 1.0])) * 5.0 * amp * rng.uniform(0.5, 1.0))
    elif kind == "edge":
        p = dict(push=7.0 * amp * u)
    elif kind == "slide":
        p = dict(yaw=sgn * 16.0 * amp * u)
    else:
        p = dict(speed=rng.uniform(0.95, 1.05))
    return Start(d, kind, p)


def apply(car, p):
    """Perturb a just-restored car in place (velocities only; the position and attitude stay as harvested)."""
    q = car.d.qvel
    if "speed" in p:
        q[:] *= p["speed"]
    if p.get("push"):
        st = car.state()
        th = st["heading"] - st["yaw"]                       # the track's direction here
        side = np.sign(st["y"]) or 1.0                      # toward the nearer bank (left is +y)
        q[0] += -np.sin(th) * side * p["push"]
        q[1] += np.cos(th) * side * p["push"]
    if p.get("yaw"):
        q[5] += p["yaw"]
    mujoco.mj_forward(car.m, car.d)


class Runner:
    """n cars driving Starts back to back (training) or a fixed list (exam / calibration)."""

    SCALE, CRASH_PEN, RES_COST = 0.3, 50.0, 0.1

    def __init__(self, base_runs, n, env_kw, seed=0):
        proto = je.JumpBatch(n, seed=seed + 7, courses=[0], **env_kw)
        self.slots = [Slot(c) for c in proto.cars]
        self.n = n
        self.base = Base(base_runs, n)
        self.cars, self.starts = [None] * n, [None] * n
        self.end_t = np.zeros(n, int)
        self.kick = [0.0] * n
        self.obs_dim = proto.obs_dim + 2

    def start(self, i, s):
        car = restore(s.drill, self.slots[i])
        apply(car, s.p)
        car.slow, car.ret = 0, 0.0
        self.kick[i] = s.p.get("kick", 0.0)
        self.cars[i], self.starts[i], self.end_t[i] = car, s, car.t + DRILL_STEPS
        self.base.set(i, s.drill.h)

    def obs(self):
        o = np.stack([c.obs() for c in self.cars]).astype(np.float32)
        self.base_a = self.base.act(o)
        return np.concatenate([o, self.base_a], 1)

    def step(self, raw):
        """raw: (n, 2) residual outputs before tanh, or None for the base car alone. Returns per-car
        (reward, crashed, ended, reason) arrays/lists; the caller restarts ended cars."""
        delta = np.zeros((self.n, 2), np.float32) if raw is None else self.SCALE * np.tanh(raw)
        a = np.clip(self.base_a + delta, -1, 1)
        rew = np.zeros(self.n, np.float32)
        crashed = np.zeros(self.n, bool)
        ended = np.zeros(self.n, bool)
        reasons = [None] * self.n
        for i, car in enumerate(self.cars):
            r, t, tc, reason = car.step(a[i], 0.0, int(self.end_t[i]), 0.0, 0.0)
            r -= self.RES_COST * float(delta[i] @ delta[i])
            if self.kick[i] and car.airborne and car.course.jump_at(car.takeoff[1]) is not None:
                car.d.qvel[4] += self.kick[i]
                self.kick[i] = 0.0
            if reason in CRASHES:
                r -= self.CRASH_PEN
                crashed[i] = True
            ended[i] = bool(t or tc)
            rew[i], reasons[i] = r, reason
        return rew, crashed, ended, reasons


def run_starts(starts, base_runs, env_kw, actor=None, n=8):
    """Drive each Start once, greedily (base + residual mean, or the base car alone). Returns crashed flags."""
    runner = Runner(base_runs, n, env_kw)
    out = [None] * len(starts)
    queue = list(range(len(starts)))
    active = [None] * n
    for i in range(n):   # cars without a start of their own drive a copy of the first, unscored
        active[i] = queue.pop(0) if queue else None
        runner.start(i, starts[active[i] if active[i] is not None else 0])
    while any(a is not None for a in active):
        x = runner.obs()
        raw = None
        if actor is not None:
            with torch.no_grad():
                raw = actor(tr._t(x)).numpy()
        _, crashed, ended, _ = runner.step(raw)
        for i in np.flatnonzero(ended):
            if active[i] is not None:
                out[active[i]] = bool(crashed[i])
            active[i] = queue.pop(0) if queue else None
            runner.start(i, starts[active[i] if active[i] is not None else 0])
    return out


def calibrate_kind(pool, kind, base_runs, env_kw, n_drills=100, iters=6, seed=0, log=print):
    """The strength at which the base car alone crashes about TARGET of the time (binary search on amp)."""
    lo, hi, hist = 0.0, 1.0, []
    rng = np.random.default_rng([seed, KINDS.index(kind)])
    for _ in range(iters):
        amp = (lo + hi) / 2
        starts = [draw(pool, kind, amp, rng) for _ in range(n_drills)]
        rate = float(np.mean(run_starts(starts, base_runs, env_kw)))
        hist.append((round(amp, 4), rate))
        log(kind, amp, rate)
        lo, hi = (amp, hi) if rate < TARGET else (lo, amp)
    best = min(hist, key=lambda h: abs(h[1] - TARGET))
    return dict(amp=best[0], rate=best[1], history=hist)


def exam_starts(pool, amps, per_kind=60, seed=12345):
    rng = np.random.default_rng(seed)
    return [draw(pool, k, amps.get(k, 0.0), rng) for k in KINDS for _ in range(per_kind)]


def exam_score(starts, crashed):
    by = collections.defaultdict(list)
    for s, c in zip(starts, crashed):
        by[s.kind].append(c)
    return dict(crash=round(float(np.mean(crashed)), 3), **{k: round(float(np.mean(v)), 3) for k, v in by.items()})


class DrillEnv2:
    """Training env: n cars, each restarted from a fresh random Start (equal mix of KINDS, calibrated strengths)."""

    def __init__(self, pool, amps, base_runs, n, env_kw, seed=0):
        self.pool, self.amps = pool, amps
        self.kinds = [k for k in KINDS if (pool["ground"] if k in ("edge", "slide") else pool[k])]
        self.rng = np.random.default_rng(seed)
        self.r = Runner(base_runs, n, env_kw, seed)
        self.n, self.obs_dim = n, self.r.obs_dim

    def _start(self, i):
        k = self.kinds[self.rng.integers(len(self.kinds))]
        self.r.start(i, draw(self.pool, k, self.amps.get(k, 0.0), self.rng))

    def reset(self):
        for i in range(self.n):
            self._start(i)
        return self.r.obs()

    def step(self, raw):
        rew, crashed, ended, reasons = self.r.step(raw)
        term = crashed
        trunc = ended & ~crashed
        infos = [{} for _ in range(self.n)]
        for i in np.flatnonzero(ended):
            infos[i] = dict(kind=self.r.starts[i].kind, reason=reasons[i], ret=self.r.cars[i].ret)
        if trunc.any():
            hs = [self.r.base.get(i) for i in range(self.n)]
            o = self.r.obs()
            for i in np.flatnonzero(trunc):
                infos[i]["terminal_observation"] = o[i]
            for i in range(self.n):
                self.r.base.set(i, hs[i])
        for i in np.flatnonzero(ended):
            self._start(i)
        return self.r.obs(), rew, term, trunc, infos


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
