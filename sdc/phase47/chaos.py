"""Phase 4.7, after E2: how chaotic are drill outcomes?

    python -m sdc.phase47.chaos      -> runs/p47-chaos.json

The base car drives each of E2's 300 exam drills REPEATS times with tiny independent action noise (std NOISE on
throttle and steering, about 1% of full range; the residual is capped at 0.3). If a drill's outcome (crash or not)
flips between repeats, it is decided by chance, not by a choice a correction could learn.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from sdc.phase47 import drills2 as d2
from sdc.phase47 import e2_prep
from sdc.phase47.train_residual import BASE, env_kw

OUT = Path("runs/p47-chaos.json")
NOISE, REPEATS = 0.02, 4
_EXAM = None


class _Noise(torch.nn.Module):
    """A stand-in 'actor' whose raw output gives a correction of about N(0, NOISE)."""

    def __init__(self, seed):
        super().__init__()
        self.g = torch.Generator().manual_seed(seed)

    def forward(self, x):
        d = torch.randn(len(x), 2, generator=self.g) * NOISE
        return torch.atanh(d / 0.3)


def _job(r):
    torch.set_num_threads(1)
    return d2.run_starts(_EXAM, BASE, env_kw(), actor=_Noise(1000 + r))


def main():
    global _EXAM
    prep = json.loads(e2_prep.OUT.read_text())
    _EXAM, _ = e2_prep.exam_set(prep["amps"])
    with mp.get_context("fork").Pool(4) as pool:
        runs = np.array(pool.map(_job, range(REPEATS)))           # (REPEATS, 300)
    base = np.array(prep["base_exam_crashed"])
    allr = np.vstack([base, runs])                                   # the noiseless run too
    p = allr.mean(0)
    kinds = [s.kind for s in _EXAM]
    out = dict(noise=NOISE, repeats=REPEATS + 1, crash_rate=round(float(p.mean()), 3),
               always=round(float((p == 1).mean()), 3), never=round(float((p == 0).mean()), 3),
               mixed=round(float(((p > 0) & (p < 1)).mean()), 3),
               base_crashes_that_flip=round(float(((p < 1) & base).sum() / max(base.sum(), 1)), 3),
               by_kind={k: dict(mixed=round(float(np.mean([(0 < q < 1) for q, kk in zip(p, kinds) if kk == k])), 3),
                                always=round(float(np.mean([q == 1 for q, kk in zip(p, kinds) if kk == k])), 3))
                        for k in d2.KINDS},
               crashed=allr.tolist())
    OUT.write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "crashed"})


if __name__ == "__main__":
    main()
