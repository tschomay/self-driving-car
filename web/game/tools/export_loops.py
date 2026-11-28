"""Export Phase 4.6 loop courses from the self-driving-car repo for the game.

    python tools/export_loops.py --out loops            (from web/game; the sim is this repo)

Each loop is written as loops/<seed>.json: the centreline and the per-point arrays loop.LoopCourse.hfield() is built
from (heights, roughness, bank), so the game rebuilds the exact heightfield the training sim uses (src/loop.js).
A checksum of the Python heightfield is included; test/loops.mjs checks the JS rebuild against it.
Seeds: validation loops 10000-10019 and practice loops 100000-100019. The gate's test loops (46000-46019) are left out.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))   # the repo root
ap.add_argument("--out", default="loops")
ap.add_argument("--seeds", nargs="*", type=int)
a = ap.parse_args()
sys.path.insert(0, str(Path(a.repo).resolve()))
from sdc.jumps import loop as lp  # noqa: E402

seeds = a.seeds or (list(range(10000, 10020)) + list(range(100000, 100020)))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
q = lambda v, s: np.rint(np.asarray(v, float) * s).astype(np.int64).tolist()   # noqa: E731
index = []
for seed in seeds:
    c = lp.make(seed)
    hf = c.hfield()
    d = dict(seed=seed, length=c.length, n=len(c.center), scale=1e6,
             cx=q(c.center[:, 0], 1e6), cy=q(c.center[:, 1], 1e6), heading=q(c.heading, 1e6), kappa=q(c.kappa, 1e6),
             profile=q(c.profile, 1e6), rough=q(c.rough, 1e6), bank=q(c.bank, 1e6),
             jumps=[j.to_dict() | dict(start=j.start, lip=j.lip, land_start=j.land_start, end=j.end) for j in c.jumps],
             whoops=list(c.whoops),
             check=dict(sum=float(hf.astype(np.float64).sum()), n_lane=int((hf * lp.ZMAX < lp.BASE + lp.BANK_H - 1e-4).sum()),
                        probe=[[int(i), int(j), float(hf[i, j])] for i, j in np.random.default_rng(seed).integers(0, lp.N, (64, 2))]))
    (out / f"{seed}.json").write_text(json.dumps(d, separators=(",", ":")))
    index.append(dict(seed=seed, length=round(c.length, 1), jumps=[j.kind for j in c.jumps],
                      whoops=c.whoops[1] > c.whoops[0], set="validation" if seed < 46000 else "practice"))
    print(seed, round(c.length, 1), [j.kind for j in c.jumps])
(out / "index.json").write_text(json.dumps(index, separators=(",", ":")))
