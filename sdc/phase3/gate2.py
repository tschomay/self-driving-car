"""Phase 3 gate, second attempt (after the traction-handling experiments E10-E14).

    python -m sdc.phase3.gate2 --run runs/p3-g2est-s0 --official outlap|single

The candidate (fixed before any result, owner's decision 2026-11-11): the owner's PPO (sdc/phase3/my_ppo.py),
seed 0, trained on the realism 3 car with the TRUE grip as a training-time input (sim-only), lap bonus 100,
exploration-noise schedule, γ 0.997, early stopping by validation gate range; at test time it drives on a
learned 1 s window grip ESTIMATE (sdc/phase3/grip_estimator.py), never the true grip.

Test set (never used by any experiment): tracks 20000-20019, gate-range physics (grip 0.5-1.3x, mass
±35%, delay 0-120 ms, noise ≤ 4%), 5 draws per track = 100 laps, physics seed 54321. Both protocols are
measured in this one look: with an out-lap before the counted lap, and a single lap from a standing start;
--official says which one decides the verdict (≥ 85%). Robustness curves on the test tracks are published.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from sdc.phase3 import grip_estimator, report, robust
from sdc.phase3.train import gifs as lap_gifs
from sdc.racer import dynamics as dy
from sdc.racer import track as trk

GATE = 0.85


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/p3-g2est-s0")
    p.add_argument("--official", choices=["outlap", "single"], required=True)
    p.add_argument("--out", default="runs/p3-gate2")
    p.add_argument("--report", default="reports/phase3-gate2.html")
    a = p.parse_args()
    import torch
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    pol, m = grip_estimator.load(a.run)
    car = grip_estimator.car_options(m["kind"])
    tests = {k: robust.gate_laps(pol, seeds=trk.TEST_SEEDS, seed=robust.TEST_PHYSICS_SEED, outlap=(k == "outlap"), **car)
             for k in ("outlap", "single")}
    test_nominal = robust.laps(pol, seeds=trk.TEST_SEEDS, **car)
    test_sweep = robust.sweep(pol, seeds=trk.TEST_SEEDS, **car)
    hand = robust.gate_laps(dy.DynamicPursuit(), seeds=trk.TEST_SEEDS, seed=robust.TEST_PHYSICS_SEED, **car)
    val = dict(single=m["gate_laps"]["completion"], outlap=m["outlap"]["gate_laps"]["completion"])
    score = tests[a.official]["completion"]
    ok = score >= GATE
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = dict(run=a.run, official=a.official, passed=ok, test={k: v for k, v in tests.items()}, test_nominal=test_nominal,
               test_sweep=test_sweep, hand_coded_test_gate=hand["completion"], validation=val,
               wall_s=round(time.perf_counter() - t0))
    (out / "gate.json").write_text(json.dumps(res, indent=1, default=float))
    other = "single" if a.official == "outlap" else "outlap"
    name = dict(outlap="with an out-lap first", single="one lap from a standing start")
    notes = [f"Gate verdict: {'PASS' if ok else 'FAIL'}.",
             f"Official protocol ({name[a.official]}): {tests[a.official]['finished']}/100 = {score:.0%} on the test set "
             f"(tracks 20000-20019, never used before, gate-range physics). Threshold 85%.",
             f"Other protocol ({name[other]}): {tests[other]['completion']:.0%}.",
             f"Same car on the validation set: {val[a.official]:.0%} ({name[a.official]}), {val[other]:.0%} ({name[other]}).",
             f"Test tracks at nominal physics: {round(20 * test_nominal['completion'])}/20. Hand-coded driver (knows the grip) "
             f"on the same 100 test laps (single lap): {hand['completion']:.0%}.",
             "Car: the owner's PPO, seed 0 (fixed in advance), trained with the true grip as a sim-only input, lap bonus, "
             "early stopping; driving on a learned 1 s grip estimate, never the true grip."]
    print("\n".join(notes))
    shown = out / "report-view"
    shown.mkdir(parents=True, exist_ok=True)
    mv = dict(m, final_eval=test_nominal, gate_laps=tests[a.official], sweep=test_sweep, label=m["label"] + " · test set")
    mv["compare"] = dict(m["compare"], gate=ok, gate_completion=score)
    mv.pop("train_range_laps", None)
    mv["gifs"] = lap_gifs(pol, shown, mv["label"], test_nominal, tracks=[trk.TEST_SEEDS[0]], **car)
    (shown / "metrics.json").write_text(json.dumps(mv, indent=1, default=float))
    report.build([shown], "Phase 3 Gate (second attempt)", "Does the owner's PPO, driving on a learned grip estimate, "
                 "finish at least 85% of laps on never-seen tracks with physics wider than it trained on?", a.report,
                 notes=notes, gifs_per_group=2)
    print(f"Report: {a.report}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
