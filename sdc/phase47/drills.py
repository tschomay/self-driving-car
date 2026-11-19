"""Phase 4.7: drills. Short episodes started in hard states, for residual reinforcement learning.

Harvest: the frozen base car (an average of Phase 4.6 students, each with its own GRU memory) drives practice loops
at training-range physics. Every SNAP_EVERY steps each car's full state is saved (physics, sensors, filter, each
student's memory, as Phase 4.6's hard.py does). Saved states become drill starts of three kinds:

  corner    the car is on a straight-ish piece (|curvature| < 0.08 /m) and a sharp turn (> 0.15 /m) starts 3-10 m
            ahead; the drill raises its speed 10-40% (every joint's velocity scaled together, wheels included).
  takeoff   a kicker's lip is 4-9 m ahead; the drill changes speed by -25%..+25%, and at the first moment the car
            is airborne it adds a pitch-rate kick of 0.5-2 rad/s, nose up or down.
  crash     1, 2 or 3 s before a place where the base car crashed (off the lane, rolled over, stuck); speed x0.95-1.05.

A drill runs DRILL_STEPS (3 s). The env (DrillEnv) drives n cars, each restarted from a random drill when its drill
ends. The policy being trained only adds a correction to the base car's action; the base car's memory runs along.
"""

from __future__ import annotations

import collections
import copy
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from sdc.jumps import env as je
from sdc.phase4 import train as tr
from sdc.phase46.ensemble import Ensemble
from sdc.phase46.hard import _FULL, snapshot

SNAP_EVERY = 25          # 0.5 s
DRILL_STEPS = 150        # 3 s
KINDS = ("corner", "takeoff", "crash")
CRASHES = ("off the lane", "rolled over", "stuck")


@dataclass
class Drill:
    kind: str
    car: object          # snapshot car (hard.snapshot)
    state: np.ndarray    # MuJoCo state vector
    h: list              # per base student: memory before this step, (n_mem, H)


class Base:
    """The frozen base car, batched, with memory that can be read and written per car."""

    def __init__(self, runs, n):
        self.ens = Ensemble(runs)
        self.nets = [m.net for m in self.ens.members]
        self.h = [net.init_memory(n) for net in self.nets]          # (n_mem, n, H) each
        self.started = torch.ones(n)

    def act(self, obs):
        acts = []
        with torch.no_grad():
            o = tr._t(obs)
            for k, net in enumerate(self.nets):
                a_in, _, self.h[k] = net.step(o, self.h[k], self.started)
                acts.append(np.tanh(net.actor(a_in).numpy()))
        self.started = torch.zeros(len(obs))
        return np.mean(acts, axis=0).astype(np.float32)

    def get(self, i):
        return [h[:, i].clone() for h in self.h]

    def set(self, i, hs, started=False):
        for k, h in enumerate(hs):
            self.h[k][:, i] = h
        self.started[i] = 1.0 if started else 0.0


def _features(car):
    st = car.state()
    c = car.course
    s = st["x"] % c.length
    idx = lambda d: int(round(((s + d) % c.length) / c.ds)) % len(c.kappa)   # noqa: E731
    k_now = abs(float(c.kappa[idx(0.0)]))
    k_ahead = max(abs(float(c.kappa[idx(d)])) for d in np.arange(3.0, 10.01, 0.5))
    lip = min(((j.lip - s) % c.length for j in c.jumps), default=99.0)
    return k_now, k_ahead, lip


def harvest(base_runs, courses, env_kw, seed=0, n=8, max_steps=2500):
    """Drive the base car on `courses` (each once); return {kind: [Drill]} and a summary."""
    rng = np.random.default_rng(seed)
    b = je.JumpBatch(n, seed=seed, courses=list(courses), **{**env_kw, "max_steps": max_steps, "jump_bonus": 0.0})
    base = Base(base_runs, n)
    obs = b.reset()
    pool = {k: [] for k in KINDS}
    ring = [collections.deque(maxlen=7) for _ in range(n)]     # (t, Drill) every 0.5 s, the last 3.5 s
    done_courses, endings = 0, collections.Counter()
    while done_courses < len(courses):
        hs = [base.get(i) for i in range(n)]
        a = base.act(obs)
        for i, car in enumerate(b.cars):
            if car.t % SNAP_EVERY == 0 and car.t > 50:
                snap_car, state = snapshot(car)
                d = Drill("?", snap_car, state, hs[i])
                ring[i].append((car.t, d))
                k_now, k_ahead, lip = _features(car)
                if k_now < 0.08 and k_ahead > 0.15:
                    pool["corner"].append(Drill("corner", snap_car, state, hs[i]))
                elif 4.0 <= lip <= 9.0:
                    pool["takeoff"].append(Drill("takeoff", snap_car, state, hs[i]))
        obs, _, term, trunc, infos = b.step(a)
        for i in np.flatnonzero(term | trunc):
            ep = infos[i]["episode"]
            endings[ep["reason"]] += 1
            done_courses += 1
            if ep["reason"] in CRASHES:
                t_end = ep["l"]
                for t, d in ring[i]:
                    if any(abs((t_end - t) - back) <= SNAP_EVERY / 2 for back in (50, 100, 150)):
                        pool["crash"].append(Drill("crash", d.car, d.state, d.h))
            ring[i].clear()
            base.set(i, [torch.zeros_like(h) for h in hs[i]], started=True)
    rng.shuffle(pool["corner"]), rng.shuffle(pool["takeoff"])
    return pool, dict(laps=done_courses, endings=dict(endings), **{k: len(v) for k, v in pool.items()})


class Slot:
    """One reusable MuJoCo model + data for restoring drills (each car's model carries its episode's physics)."""

    def __init__(self, car):
        self.m, self.d, self.scratch = car.m, car.d, car._scratch
        self.course_seed = None


def restore(drill, slot):
    snap_car = drill.car
    memo = {id(snap_car.m): slot.m, id(snap_car.course): snap_car.course, id(snap_car._scratch): slot.scratch,
            id(snap_car.courses): snap_car.courses}
    car = copy.deepcopy(snap_car, memo)
    car.d = slot.d
    if slot.course_seed != car.course.seed:
        slot.m.hfield_data[:] = car.course.hfield().ravel()
        slot.course_seed = car.course.seed
    car._apply_phys()
    mujoco.mj_setState(slot.m, slot.d, drill.state, _FULL)
    return car


class DrillEnv:
    """n cars, each driving drills back to back. Actions are the residual policy's raw outputs; the car drives
    clip(base + SCALE * tanh(raw)). Observation: sensors + the base car's action (what the base car wants now).

    Reward per step: metres along the lane - RES_COST * |correction|^2; a crash ends the drill with CRASH_PEN.
    A drill that runs its 3 s (or finishes the lap) is truncated, not ended (the critic's value carries on)."""

    SCALE, CRASH_PEN, RES_COST = 0.3, 50.0, 0.1
    MIX = dict(corner=0.4, takeoff=0.3, crash=0.3)

    def __init__(self, pool, base_runs, n, env_kw, seed=0):
        self.pool = {k: v for k, v in pool.items() if v}
        mix = {k: w for k, w in self.MIX.items() if k in self.pool}
        self.kinds, self.p = list(mix), np.array(list(mix.values())) / sum(mix.values())
        self.rng = np.random.default_rng(seed)
        proto = je.JumpBatch(n, seed=seed + 7, courses=[0], **env_kw)
        self.slots = [Slot(c) for c in proto.cars]
        self.n = n
        self.base = Base(base_runs, n)
        self.cars = [None] * n
        self.end_t = np.zeros(n, int)
        self.kick = [0.0] * n
        self.kind = [""] * n
        self.obs_dim = proto.obs_dim + 2

    def _start(self, i):
        kind = self.kinds[self.rng.choice(len(self.kinds), p=self.p)]
        d = self.pool[kind][self.rng.integers(len(self.pool[kind]))]
        car = restore(d, self.slots[i])
        f = dict(corner=(1.1, 1.4), takeoff=(0.75, 1.25), crash=(0.95, 1.05))[kind]
        car.d.qvel[:] *= self.rng.uniform(*f)
        mujoco.mj_forward(car.m, car.d)
        car.slow, car.ret = 0, 0.0
        self.kick[i] = float(self.rng.choice([-1, 1]) * self.rng.uniform(0.5, 2.0)) if kind == "takeoff" else 0.0
        self.cars[i], self.end_t[i], self.kind[i] = car, car.t + DRILL_STEPS, kind
        self.base.set(i, d.h)

    def _obs(self):
        o = np.stack([c.obs() for c in self.cars]).astype(np.float32)
        self.base_a = self.base.act(o)
        return np.concatenate([o, self.base_a], 1)

    def reset(self):
        for i in range(self.n):
            self._start(i)
        return self._obs()

    def step(self, raw):
        delta = self.SCALE * np.tanh(raw)
        a = np.clip(self.base_a + delta, -1, 1)
        rew = np.zeros(self.n, np.float32)
        term = np.zeros(self.n, bool)
        trunc = np.zeros(self.n, bool)
        infos = [{} for _ in range(self.n)]
        for i, car in enumerate(self.cars):
            r, t, tc, reason = car.step(a[i], 0.0, int(self.end_t[i]), 0.0, 0.0)
            r -= self.RES_COST * float(delta[i] @ delta[i])
            if self.kick[i] and car.airborne:
                car.d.qvel[4] += self.kick[i]
                self.kick[i] = 0.0
            if reason in CRASHES:
                r -= self.CRASH_PEN
                term[i] = True
            elif t or tc:          # time's up, or the lap finished: not a failure
                trunc[i] = True
            rew[i] = r
            if term[i] or trunc[i]:
                infos[i] = dict(kind=self.kind[i], reason=reason, ret=car.ret)
        need_term_obs = np.flatnonzero(trunc)
        if len(need_term_obs):    # the value after a truncated drill: the sensors then, plus the base car's action
            hs = [self.base.get(i) for i in range(self.n)]
            o = self._obs()
            for i in need_term_obs:
                infos[i]["terminal_observation"] = o[i]
            for i in range(self.n):
                self.base.set(i, hs[i])
        for i in np.flatnonzero(term | trunc):
            self._start(i)
        return self._obs(), rew, term, trunc, infos


class ResidualPolicy:
    """Base car + a trained residual, greedy (for dr.run / common.prun)."""

    def __init__(self, base_runs, actor_state, obs_dim):
        self.ens = Ensemble(base_runs)
        self.net = tr.ActorCritic(obs_dim=obs_dim)
        self.net.load_state_dict(actor_state)
        self.env_kw = {}

    def reset(self, n):
        self.ens.reset(n)

    def __call__(self, b):
        base_a = self.ens(b).astype(np.float32)
        x = np.concatenate([b.obs(), base_a], 1)
        with torch.no_grad():
            delta = DrillEnv.SCALE * np.tanh(self.net.actor(tr._t(x)).numpy())
        return np.clip(base_a + delta, -1, 1)


def load_residual(base_runs, path, obs_dim):
    torch.set_num_threads(1)
    return ResidualPolicy(base_runs, torch.load(path), obs_dim)
