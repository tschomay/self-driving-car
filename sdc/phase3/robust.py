"""Robustness evaluation for Phase 3: lap completion on held-out tracks as the physics changes.

sweep()      one physics knob at a time (the others nominal), one lap per held-out track per
             value: the robustness curves.
gate_laps()  physics drawn at random from a range (default the gate's, wider than training),
             DRAWS laps per held-out track, each with its own draw: the gate number.

A policy is a callable policy(batch) -> actions, as in sdc.racer.evaluate.
"""

from __future__ import annotations

import functools

import numpy as np

from sdc.racer import dynamics as dy
from sdc.racer import evaluate
from sdc.racer import track as trk

SWEEPS = dict(
    grip=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
    delay=[0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
    mass=[0.65, 0.8, 1.0, 1.2, 1.35],
    noise=[0.0, 0.01, 0.02, 0.03, 0.04, 0.06],
)
TRAIN_SPAN = {k: getattr(dy.TRAIN_DR, k) for k in SWEEPS}
GATE_SPAN = {k: getattr(dy.GATE_DR, k) for k in SWEEPS}
DRAWS = 5


def laps(policy, physics="nominal", dirt=True, seeds=trk.HELD_OUT_SEEDS, record=False, seed=0, setup=None,
         realism=2):
    mk = functools.partial(dy.DynamicRacerBatch, physics=physics, dirt=dirt, seed=seed, realism=realism)
    return evaluate.run_laps(policy, seeds, record=record, make_batch=mk, setup=setup)


def sweep(policy, knobs=("grip", "delay", "mass", "noise"), dirt=True, realism=2) -> dict:
    out = {}
    for k in knobs:
        rows = []
        for v in SWEEPS[k]:
            r = laps(policy, physics={k: v}, dirt=dirt, realism=realism)
            rows.append(dict(value=v, completion=r["completion"], mean_lap_time=r["mean_lap_time"]))
        out[k] = rows
    return out


def gate_laps(policy, physics=dy.GATE_DR, draws=DRAWS, dirt=True, seed=12345, realism=2) -> dict:
    """DRAWS laps per held-out track, each lap with physics drawn from `physics`. Reproducible."""
    rows = []
    for d in range(draws):
        r = laps(policy, physics=physics, dirt=dirt, seed=seed + d, realism=realism)
        for t in r["tracks"]:
            rows.append(dict(t, draw=d))
    done = [r["finished"] for r in rows]
    return dict(tracks=rows, completion=float(np.mean(done)), n=len(rows), finished=int(np.sum(done)),
                physics=dy.physics_range(physics).to_dict())
