"""Phase 3 E13b: a careful out-lap by rule (no retraining).

During the out-lap (until the car has covered one lap's distance), the policy's throttle is capped at
`cap` (braking untouched); the grip estimator keeps learning the conditions meanwhile. The counted lap is
driven normally.

    python -m sdc.phase3.outlap runs/p3-e13window-s0 runs/p3-e13drive-s0 ... --caps 0.4 0.6 0.8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sdc.phase3 import grip_estimator, robust


def careful_outlap(policy, cap):
    def act(b):
        a = np.array(policy(b), dtype=float)
        out = b.progress < b.length
        a[:, 1] = np.where(out & (a[:, 1] > cap), cap, a[:, 1])
        return a
    return act


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--caps", type=float, nargs="+", default=[0.4, 0.6, 0.8])
    a = p.parse_args()
    for d in a.runs:
        pol, m = grip_estimator.load(d)
        car = grip_estimator.car_options(m["kind"])
        res = {}
        for cap in a.caps:
            g = robust.gate_laps(careful_outlap(pol, cap), outlap=True, **car)
            rows = g["tracks"]
            fails = [r for r in rows if not r["finished"]]
            survived = [r for r in rows if r["laps"] >= 0.5]
            res[str(cap)] = dict(gate=g["completion"], fail_outlap=sum(r["laps"] < 0.5 for r in fails),
                                 counted_after_outlap=float(np.mean([r["finished"] for r in survived])),
                                 lap=float(np.mean([r["lap_time"] for r in rows if r["finished"]])),
                                 tracks=rows)
            print(d, cap, {k: v for k, v in res[str(cap)].items() if k != "tracks"}, flush=True)
        m["careful_outlap"] = res
        (Path(d) / "metrics.json").write_text(json.dumps(m, indent=1, default=float))


if __name__ == "__main__":
    main()
