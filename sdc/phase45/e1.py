"""E1: what breaks? Phase 4's gate car and the experts, unchanged, on the realistic car.

  python -m sdc.phase45.e1 --run runs/p4-e5-fb-s2 --out runs/p45-e1

Every config is 20 validation courses x 3 starts (180 jumps), one car at a time, 1 torch thread.
Each physics switch on its own, the noisy sensors on their own, then everything together at nominal,
training-range and gate-range physics. Writes <out>/e1.json.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from sdc.jumps import course as crs
from sdc.jumps import driver as dr
from sdc.phase45.common import EXPERT, EXPERT_P4

CONFIGS = {
    "Phase 4 car (as at the gate)": {},
    "wide tires": dict(real=("tires",)),
    "soft suspension": dict(real=("susp",)),
    "one motor + diffs": dict(real=("drive",)),
    "servo lag": dict(real=("servo",)),
    "battery sag": dict(real=("battery",)),
    "control delay": dict(real=("delay",)),
    "realistic sensors": dict(sensors="noisy"),
    "all physics": dict(real="all"),
    "all physics + sensors": dict(real="all", sensors="noisy"),
    "... + training-range physics": dict(real="all", sensors="noisy", dr="train"),
    "... + gate-range physics": dict(real="all", sensors="noisy", dr="gate"),
}
RUN = None


def job(args):
    who, cfg = args
    import torch
    torch.set_num_threads(1)
    if who == "gate car":
        from sdc.phase4 import train as tr
        pol, _ = tr.load(Path(RUN))
    else:
        pol = dr.ExpertDriver(**(EXPERT if who == "expert (4.5)" else EXPERT_P4))
    eps = dr.run(pol, crs.VAL_SEEDS, n_envs=1, starts=dr.STARTS, **CONFIGS[cfg])
    return who, cfg, dr.score(eps), eps


def _init(run):
    global RUN
    RUN = run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/p4-e5-fb-s2")
    ap.add_argument("--out", default="runs/p45-e1")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    jobs = [(w, c) for c in CONFIGS for w in ("gate car", "expert (Phase 4)", "expert (4.5)")]
    out = {}
    with Pool(a.procs, initializer=_init, initargs=(a.run,)) as p:
        for who, cfg, sc, eps in p.imap_unordered(job, jobs):
            out.setdefault(cfg, {})[who] = dict(score=sc, episodes=eps)
            print(f"{cfg:32s} {who:18s} {sc['rate']:.1%}  {sc['by_kind']}  {sc['reasons']}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, "e1.json").write_text(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
