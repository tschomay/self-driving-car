"""Phase 3.5 gate.

    python -m sdc.phase35.gate --run runs/p35-e5-lo-s0

Candidate (fixed before any result, owner's go 2026-11-12): E9 exactly. The owner's network from E4 (behaviour
clone of the hand-coded driver, then the owner's PPO at low noise, seed 0, fastest checkpoint with >= 90%
validation gate range), driving with no oracle input: dense brake-pulse probes on the out-lap (every 3 m,
corners allowed, total force, K from training-range physics), a 2.5 m odometer map that takes the lowest
reading within ±2.5 m, odometer position reset at the start line.

Test set (never used before): tracks 30000-30019 (trk.TEST35_SEEDS), physics seed 67890
(robust.TEST35_PHYSICS_SEED), gate-range physics, 5 draws per track = 100 attempts. Protocol: an out-lap, then
the timed lap; both must be finished.
Pass: >= 85% finished at gate range AND nominal timed lap <= 1.2x the hand-coded driver's timed lap on the same
test tracks and protocol.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sdc.phase3 import robust
from sdc.phase35 import probe
from sdc.phase35 import selfmap as sm
from sdc.racer import dynamics as dy
from sdc.racer import track as trk

GATE, RATIO = 0.85, 1.2


def candidate(run):
    net, base = sm.load_net(run)
    return probe.ProbeDriver(net, base, corners=True, every=3.0, bin=2.5, erode=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/p35-e5-lo-s0")
    ap.add_argument("--out", default="runs/p35-gate")
    a = ap.parse_args()
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    seeds, ps = trk.TEST35_SEEDS, robust.TEST35_PHYSICS_SEED
    gate = robust.gate_laps(candidate(a.run), seeds=seeds, seed=ps, outlap=True, **sm.CAR)
    nominal = robust.laps(candidate(a.run), seeds=seeds, outlap=True, **sm.CAR)
    hand_nom = robust.laps(dy.DynamicPursuit(), seeds=seeds, outlap=True, **sm.CAR)
    hand_gate = robust.gate_laps(dy.DynamicPursuit(), seeds=seeds, seed=ps, outlap=True, **sm.CAR)
    bar = RATIO * hand_nom["mean_lap_time"]
    lap = nominal["mean_lap_time"]
    ok_rel = gate["completion"] >= GATE
    ok_speed = lap is not None and lap <= bar
    res = dict(run=a.run, passed=bool(ok_rel and ok_speed), gate=gate, nominal=nominal, hand_nominal=hand_nom,
               hand_gate_completion=hand_gate["completion"], bar=bar, ratio=(lap / hand_nom["mean_lap_time"]) if lap else None,
               reliability_pass=ok_rel, speed_pass=ok_speed, wall_s=round(time.perf_counter() - t0))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"GATE {'PASS' if res['passed'] else 'FAIL'}: gate range {gate['completion']:.0%} (>= 85%: {ok_rel}); "
          f"nominal {nominal['completion']:.0%}, lap {lap} vs bar {bar:.2f} ({res['ratio']}); "
          f"hand-coded {hand_nom['mean_lap_time']:.2f} s, gate range {hand_gate['completion']:.0%}")


if __name__ == "__main__":
    main()
