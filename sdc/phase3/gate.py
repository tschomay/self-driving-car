"""Phase 3 gate: the owner's PPO on the TEST set, with physics wider than training.

    python -m sdc.phase3.gate                       # train the gate policy, then evaluate (about 15 min)
    python -m sdc.phase3.gate --run runs/p3-e9-s0   # evaluate an already-trained run instead

Gate (docs/roadmap.md, owner's decisions at Phase 3 entry and 2026-11-10):
  - policy: the owner's PPO (sdc/phase3/my_ppo.py) with E8's recipe: full train-range randomization
    (grip 0.7-1.1x, mass ±20%, delay 0-60 ms, noise ≤ 2%), accelerometer, γ 0.997, 1M steps, seed 0
    (fixed before any result was seen)
  - test set: tracks 20000-20019, never used by any experiment, physics drawn from the gate range
    (grip 0.5-1.3x, mass ±35%, delay 0-120 ms, noise ≤ 4%), 5 draws per track = 100 laps, physics seed 54321
  - pass: ≥ 85% of those 100 laps finished; robustness curves (on the test tracks) published
The same numbers on the validation set (tracks 10000-10019) are reported alongside, to show how much
choosing recipes on it flattered them. Writes reports/phase3-gate.html and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from sdc.phase3 import my_ppo, report, robust
from sdc.phase3.train import car_of
from sdc.phase3.train import gifs as lap_gifs
from sdc.racer import dynamics as dy
from sdc.racer import track as trk

GATE = 0.85


def gifs(pol, out, label, final, car):
    """Lap GIFs on the first test track and the first failed one (nominal physics)."""
    return lap_gifs(pol, out, label, final, tracks=[trk.TEST_SEEDS[0]], **car)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="evaluate this trained run instead of training one")
    p.add_argument("--out", default="runs/p3-gate")
    p.add_argument("--report", default="reports/phase3-gate.html")
    a = p.parse_args()
    t0 = time.perf_counter()
    run = Path(a.run or a.out)
    if a.run is None:
        my_ppo.train(0, 1_000_000, run, label="my PPO, γ 0.997 (gate)", gamma=0.997)
    pol, m = my_ppo.load_policy(run)
    car = car_of(m["env"])
    test = robust.gate_laps(pol, seeds=trk.TEST_SEEDS, seed=robust.TEST_PHYSICS_SEED, **car)
    test_nominal = robust.laps(pol, seeds=trk.TEST_SEEDS, **car)
    test_sweep = robust.sweep(pol, seeds=trk.TEST_SEEDS, **car)
    hand = robust.gate_laps(dy.DynamicPursuit(), seeds=trk.TEST_SEEDS, seed=robust.TEST_PHYSICS_SEED, **car)
    val = m["gate_laps"]["completion"]
    ok = test["completion"] >= GATE
    out = dict(run=str(run), test_gate_laps=test, test_nominal=test_nominal, test_sweep=test_sweep,
               hand_coded_test_gate=hand["completion"], validation_gate=val, passed=ok,
               wall_s=round(time.perf_counter() - t0))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "gate.json").write_text(json.dumps(out, indent=1, default=float))
    notes = [f"Gate verdict: {'PASS' if ok else 'FAIL'}.",
             f"Test set (tracks 20000-20019, never used before, gate-range physics, 100 laps): "
             f"{test['finished']}/100 = {test['completion']:.0%} (threshold 85%).",
             f"Same policy on the validation set (tracks 10000-10019, used by every experiment): {val:.0%}.",
             f"Test tracks at nominal physics: {round(20 * test_nominal['completion'])}/20. "
             f"Hand-coded driver (knows grip) on the same 100 test laps: {hand['completion']:.0%}.",
             f"Policy: {m['label']}, seed {m['seed']}, {m['trained_timesteps']:,} steps. Whole command: "
             f"{out['wall_s'] / 60:.0f} min."]
    print("\n".join(notes))
    # the report page shows the TEST numbers: a copy of the run's metrics with the test results swapped in
    shown = Path(a.out) / "report-view"
    shown.mkdir(parents=True, exist_ok=True)
    mv = dict(m, final_eval=test_nominal, gate_laps=test, sweep=test_sweep, label=m["label"] + " · test set")
    mv["compare"] = dict(m["compare"], gate=ok, gate_completion=test["completion"])
    mv.pop("train_range_laps", None)
    mv["gifs"] = gifs(pol, shown, mv["label"], test_nominal, car)
    (shown / "metrics.json").write_text(json.dumps(mv, indent=1, default=float))
    report.build([shown], "Phase 3 Gate", "Does the owner's PPO finish at least 85% of laps on never-seen tracks "
                 "with physics wider than it trained on?", a.report, notes=notes, gifs_per_group=2)
    print(f"Report: {a.report}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
