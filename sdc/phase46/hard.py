"""Phase 4.6 E8: more copying where the students disagree (owner's idea, 2026-11-18).

    python -m sdc.phase46.hard --out runs/p46-e8-hard          # replay hard spots
    python -m sdc.phase46.hard --out runs/p46-e8-plain --plain # control: the same amount of ordinary copying

Starts from E5's three students (copies of the cautious teacher) and keeps copying the same teacher, with the
three students' average driving (as the gate car does), so the data comes from where the gate car goes.

Each round collects STEPS steps on practice loops (training-range physics). At every step the three students each
propose an action; their disagreement is the largest spread (standard deviation) of steering or throttle. In the
hard arm, when the disagreement passes the threshold (the top 5% of the first round's ordinary driving), the car is
copied as it was about 1 s earlier (physics, sensors, filter, and each student's memory) and K branches re-drive
the next 2.6 s from there with extra random noise on the actions. The cautious teacher labels every step, main and
branch. Up to half of each round's steps come from branches; the control fills the same budget with ordinary laps.

Every student is then refitted (bc_rnn.fit) on all data so far, from its own memory along each piece of road.
Scored as E7: the average of the three on validation loops x 3 starts, gate-range physics; also each alone.
"""

from __future__ import annotations

import argparse
import collections
import copy
import functools
import json
import pickle
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from sdc.jumps import env as je
from sdc.phase4 import my_gru
from sdc.phase4 import train_rnn as trr
from sdc.phase45 import bc_rnn
from sdc.phase46 import common as cm
from sdc.phase46.bc_loop import REF, TRAIN_MAP, student_env
from sdc.phase46.ensemble import make
from sdc.phase46.teacher import LoopExpert
from sdc.phase4 import train as tr

E5 = [f"runs/p46-e5-s{s}" for s in (0, 1, 2)]
N_CARS = 16
SNAP_EVERY, SNAP_KEEP = 10, 6          # a snapshot every 0.2 s; the oldest is about 1.2 s back
K, BRANCH_LEN = 4, 128                 # 4 branches of 2.6 s (two 64-step chunks) from each hard spot
NOISE, BRANCH_NOISE = 0.1, 0.25
COOLDOWN = 100                         # one hard spot per car per 2 s


_FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS | mujoco.mjtState.mjSTATE_WARMSTART | mujoco.mjtState.mjSTATE_CTRL
_POOL: dict = {}


def snapshot(car):
    """The car's state: everything but MuJoCo's data is deep-copied (sharing the model, course and scratch buffers,
    which don't change within a lap); MuJoCo's data is kept as its state vector. Copying MjData itself leaked memory."""
    memo = {id(car.m): car.m, id(car.course): car.course, id(car._scratch): car._scratch,
            id(car.courses): car.courses, id(car.d): None}
    snap = copy.deepcopy(car, memo)
    state = np.empty(mujoco.mj_stateSize(car.m, _FULL))
    mujoco.mj_getState(car.m, car.d, state, _FULL)
    return snap, state


def restore(snap, slot):
    """A drivable car from a snapshot, using a reused MjData buffer (one per branch slot)."""
    snap_car, state = snap
    memo = {id(snap_car.m): snap_car.m, id(snap_car.course): snap_car.course,
            id(snap_car._scratch): snap_car._scratch, id(snap_car.courses): snap_car.courses}
    car = copy.deepcopy(snap_car, memo)
    key = (id(car.m), slot)
    if key not in _POOL:
        _POOL.clear() if len(_POOL) >= 32 * K else None
        _POOL[key] = mujoco.MjData(car.m)
    car.d = _POOL[key]
    mujoco.mj_setState(car.m, car.d, state, _FULL)
    mujoco.mj_forward(car.m, car.d)
    return car


class Recorder:
    """Per student: (obs, teacher action, started, memory-before) along each car's road, cut into chunks."""

    def __init__(self, n_students, L):
        self.L = L
        self.data = [[] for _ in range(n_students)]

    def add(self, O, A, S, Hs):
        """O, A, S: (T, n, ...); Hs[k]: (T, n_mem, n, H)."""
        for k, H in enumerate(Hs):
            self.data[k] += bc_rnn.chunks(O, A, S, H, self.L)


def students_act(nets, o, hs, started):
    acts, new = [], []
    with torch.no_grad():
        for net, h in zip(nets, hs):
            a_in, _, h2 = net.step(tr._t(o), h, started)
            acts.append(np.tanh(net.actor(a_in).numpy()))
            new.append(h2)
    acts = np.stack(acts)                              # (k, n, 2)
    return acts, new


def branch(cars, hs, teacher, nets, rng, steps):
    """Re-drive copies of cars (all from one snapshot) for `steps` steps. Returns arrays for Recorder.add, cut to
    the steps before the first car ends (so no car resets, which would change the shared model)."""
    n = len(cars)
    started = torch.zeros(n)
    O, A, S, H = [], [[] for _ in nets], [], None
    Hs = [[] for _ in nets]
    for _ in range(steps):
        o = np.stack([c.obs() for c in cars]).copy()
        a_exp = np.stack([teacher.act(c) for c in cars])
        for k in range(len(nets)):
            Hs[k].append(hs[k].clone())
        acts, hs = students_act(nets, o, hs, started)
        a = np.clip(acts.mean(0) + BRANCH_NOISE * rng.standard_normal((n, 2)), -1, 1)
        O.append(o)
        A[0].append(a_exp.astype(np.float32))
        S.append(started.numpy().copy())
        done = False
        for i, c in enumerate(cars):
            _, term, trunc, _ = c.step(a[i], 10.0, 10**9, 0.03, 0.0)
            done |= bool(term or trunc)
        if done:
            break
    return np.stack(O), np.stack(A[0]), np.stack(S), [torch.stack(h) for h in Hs]


def collect(nets, teacher, n_steps, seed, hard, thresh, rec):
    """One round. Returns (main steps, branch steps, disagreement values seen in main driving, hard spots)."""
    b = je.JumpBatch(N_CARS, seed=seed, **{k: v for k, v in bc_rnn.ENV.items() if k not in ("val_dr", "val_map")})
    b.reset()
    rng = np.random.default_rng(seed)
    hs = [net.init_memory(N_CARS) for net in nets]
    started = torch.ones(N_CARS)
    ring = [collections.deque(maxlen=SNAP_KEEP) for _ in range(N_CARS)]
    cool = np.zeros(N_CARS, int)
    O, A, S, Hs, D = [], [], [], [[] for _ in nets], []
    main_steps = branch_steps = spots = 0
    budget = n_steps // 2 if hard else 0
    while main_steps + branch_steps < n_steps:
        o = b.obs().copy()
        a_exp = teacher(b)
        for k in range(len(nets)):
            Hs[k].append(hs[k].clone())
        acts, hs = students_act(nets, o, hs, started)
        d = acts.std(0).max(1)                         # per car: the larger spread of steering or throttle
        D.append(d)
        a = np.clip(acts.mean(0) + NOISE * rng.standard_normal((N_CARS, 2)), -1, 1)
        S.append(started.numpy().copy())
        O.append(o)
        A.append(a_exp.astype(np.float32))
        _, _, term, trunc, _ = b.step(a)
        main_steps += N_CARS
        started = torch.as_tensor((term | trunc).astype(np.float32))
        cool -= 1
        for i, c in enumerate(b.cars):
            if term[i] or trunc[i]:
                ring[i].clear()
                continue
            if c.t % SNAP_EVERY == 0:
                ring[i].append((snapshot(c), [h[:, i:i + 1].clone() for h in hs]))
            if hard and branch_steps < budget and d[i] > thresh and cool[i] <= 0 and len(ring[i]) == SNAP_KEEP:
                snap, h0 = ring[i][0]
                cars = [restore(snap, j) for j in range(K)]
                hb = [h.repeat(1, K, 1) for h in h0]
                bo, ba, bs, bh = branch(cars, hb, teacher, nets, rng, BRANCH_LEN)
                rec.add(bo, ba, bs, bh)
                branch_steps += bo.shape[0] * K
                spots += 1
                cool[i] = COOLDOWN
    rec.add(np.stack(O), np.stack(A), np.stack(S), [torch.stack(h) for h in Hs])
    return main_steps, branch_steps, np.concatenate(D), spots


def _fit(args):
    k, path, data, epochs, seed, obs_dim = args
    torch.set_num_threads(1)
    net = trr.RecurrentActorCritic(obs_dim=obs_dim)
    net.load_state_dict(torch.load(path))
    mse = bc_rnn.fit(net, data, epochs, lr=3e-4, seed=seed + k)
    torch.save(net.state_dict(), path)
    return mse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plain", action="store_true", help="control: no hard-spot branches")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--jump-v", default="{}", help='E9: teacher approach speed per jump kind, JSON, e.g. {"drop": 6.0}')
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    t0 = time.perf_counter()
    out = Path(a.out)
    runs = [out / f"s{k}" for k in range(3)]
    maps = json.loads(Path(TRAIN_MAP).read_text())
    courses = [c for c in cm.PRACTICE if maps[f"{c}|0.00"]["kappa"]]
    env = student_env(False)
    bc_rnn.ENV.clear()
    bc_rnn.ENV.update(env, steer_cost=0.03, jump_bonus=10.0, courses=courses)
    obs_dim = je.JumpBatch(1, **env).obs_dim
    nets = []
    for src, dst in zip(E5, runs):
        dst.mkdir(parents=True, exist_ok=True)
        m = json.loads((Path(src) / "metrics.json").read_text())
        net = trr.RecurrentActorCritic(obs_dim=obs_dim)
        net.load_state_dict(torch.load(Path(src) / "model.pt"))
        torch.save(net.state_dict(), dst / "model.pt")
        (dst / "metrics.json").write_text(json.dumps(dict(obs_dim=obs_dim, from_run=src, teacher=m["teacher"])))
        nets.append(net)
    jump_v = json.loads(a.jump_v)
    teacher = LoopExpert(a_lat=2.5, a_brake=2.5, jump_v=jump_v)
    rec = Recorder(3, my_gru.CONFIG["seq_len"])
    val_env = student_env(False, val=True)
    ref = pickle.load(open(REF, "rb"))
    thresh, log = None, []
    from multiprocessing import Pool
    for r in range(a.rounds):
        hard = not a.plain and thresh is not None
        ms, bs, D, spots = collect(nets, teacher, a.steps, a.seed * 1000 + 500 + r, hard, thresh or 1e9, rec)
        if thresh is None:
            thresh = float(np.quantile(D, 0.95))
        with Pool(min(a.procs, 3)) as p:
            mses = p.map(_fit, [(k, str(runs[k] / "model.pt"), rec.data[k], a.epochs, a.seed, obs_dim)
                                for k in range(3)])
        for k, net in enumerate(nets):
            net.load_state_dict(torch.load(runs[k] / "model.pt"))
        eps = cm.prun(functools.partial(make, tuple(str(x) for x in runs)), cm.VAL_SEEDS, starts=(0.0,),
                      procs=a.procs, **val_env)
        sc = cm.lap_score(eps, teacher=ref[::3])
        log.append(dict(round=r, main_steps=ms, branch_steps=bs, hard_spots=spots, thresh=round(thresh, 4),
                        disagree_mean=round(float(D.mean()), 4), disagree_p95=round(float(np.quantile(D, 0.95)), 4),
                        mse=[round(x, 4) for x in mses], finished=sc["finished_n"], landed=sc["rate"],
                        wall_s=round(time.perf_counter() - t0)))
        print(log[-1], flush=True)
    eps = cm.prun(functools.partial(make, tuple(str(x) for x in runs)), cm.VAL_SEEDS, procs=a.procs, **val_env)
    final = cm.lap_score(eps, teacher=ref)
    alone = []
    for k in range(3):
        e = cm.prun(functools.partial(cm.load_student, str(runs[k]), obs_dim), cm.VAL_SEEDS, procs=a.procs, **val_env)
        alone.append(cm.lap_score(e, teacher=ref)["in_time_n"])
    m = dict(label="E5 students, more copying" + (" (control)" if a.plain else " at hard spots"), plain=a.plain,
             seed=a.seed, jump_v=jump_v, rounds=log, final=final, final_episodes=eps, alone=alone, env46=val_env,
             wall_s=round(time.perf_counter() - t0, 1))
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    print(f"final: average finished {final['finished_n']}/60, in time {final['in_time_n']}/60, landed "
          f"{final['rate']:.0%}, endings {final['reasons']}; alone {alone}")


if __name__ == "__main__":
    main()
