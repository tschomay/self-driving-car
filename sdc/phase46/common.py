"""Shared Phase 4.6 settings and helpers."""

from __future__ import annotations

from multiprocessing import Pool

import numpy as np

from sdc.jumps import driver as dr
from sdc.jumps import loop as lp

VAL_SEEDS = lp.VAL_SEEDS
TEST_SEEDS = lp.TEST_SEEDS               # gate only: loops 46000-46019
PHYS_SEED_VAL = 0
PHYS_SEED_TEST = 46                      # physics/noise stream for the gate (never used by experiments)
PRACTICE = list(range(100000, 100300))
MAX_STEPS = 3000                         # 60 s: a timed lap takes about 30 s
CAR = dict(real="all", sensors="parts", attitude=True, loop=True, max_steps=MAX_STEPS)   # Phase 4.5's realistic car
TIME_FACTOR = 1.5                        # gate: a lap counts if finished within 1.5x the teacher's time

_MAKE = None


def _init(make):
    global _MAKE
    import torch
    torch.set_num_threads(1)
    _MAKE = make


def _job(args):
    course, starts, kw = args
    return dr.run(_MAKE(), [course], n_envs=1, starts=starts, **kw)


def prun(make_policy, courses, starts=dr.STARTS, procs=4, **env_kw):
    """dr.run in parallel, one course per job, one car at a time (scores don't depend on batching).
    make_policy: a picklable zero-argument function that builds the policy in each worker."""
    jobs = [(c, starts, env_kw) for c in courses]
    with Pool(procs, initializer=_init, initargs=(make_policy,)) as p:
        return [e for eps in p.map(_job, jobs, chunksize=1) for e in eps]


def lap_score(episodes, teacher=None):
    """Finish rate, and (given the teacher's episodes on the same runs, same order) the gate's finish: within
    TIME_FACTOR x the teacher's time. A run the teacher didn't finish is timed against the teacher's median lap."""
    sc = dr.score(episodes)
    sc["finished_n"] = int(sum(e["finished"] for e in episodes))
    sc["runs"] = len(episodes)
    sc["lap_s"] = round(float(np.median([e["l"] for e in episodes if e["finished"]] or [np.nan])) * 0.02, 2)
    if teacher is not None:
        med = float(np.median([t["l"] for t in teacher if t["finished"]]))
        ok = [e["finished"] and e["l"] <= TIME_FACTOR * (t["l"] if t["finished"] else med)
              for e, t in zip(episodes, teacher)]
        sc["in_time_n"] = int(sum(ok))
        sc["in_time"] = round(float(np.mean(ok)), 3)
    return sc


def load_student(run, obs_dim=None):
    """A Phase 4.6 student (bc_loop.py): the owner's GRU network, greedy. Env options come from the caller
    (training's are in metrics.json under env46)."""
    import json
    from pathlib import Path

    import torch

    from sdc.phase4 import train_rnn as trr
    if obs_dim is None:
        obs_dim = json.loads((Path(run) / "metrics.json").read_text())["obs_dim"]
    net = trr.RecurrentActorCritic(obs_dim=obs_dim)
    net.load_state_dict(torch.load(Path(run) / "model.pt"))
    return trr.GreedyRNN(net, {})
