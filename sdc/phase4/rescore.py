"""Re-score every checkpoint of a run from 3 starts (180 validation jumps) -> <run>/rescore.json.

The in-training evaluation drives each validation course once (60 jumps). Landings are chaotic,
so that number is noisy and picking the best of 20 checkpoints by it flatters the pick.
Usage: python -m sdc.phase4.rescore runs/p4-e1-s0 [--rnn]
"""

import argparse
import json
from pathlib import Path

import torch

from sdc.jumps import course as crs
from sdc.jumps import driver as dr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--rnn", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    d = Path(a.run)
    if a.rnn:
        from sdc.phase4 import train_rnn as mod
    else:
        from sdc.phase4 import train as mod
    pol, m = mod.load(d)
    out = []
    for ck in sorted((d / "ckpt").glob("*.pt"), key=lambda q: int(q.stem)):
        pol.net.load_state_dict(torch.load(ck))
        if hasattr(pol, "h"):
            pol.h = None
        sc = dr.score(dr.run(pol, crs.VAL_SEEDS, starts=dr.STARTS))
        out.append(dict(timesteps=int(ck.stem), rate=sc["rate"], finished=sc["finished"]))
        print(out[-1], flush=True)
    (d / "rescore.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
