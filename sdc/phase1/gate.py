"""Phase 1 gate: the owner's own PPO must drive tracks it has never seen.

    python -m sdc.phase1.gate

Gate (docs/roadmap.md; the owner decided it counts for their own PPO only): trained with
sdc/phase1/my_ppo.py (textbook clipping, the owner's choice after E5), the policy finishes
≥ 95% of laps on the 20 held-out tracks (seeds 10000-10019, never trained on) with mean
lap time ≤ 1.1× the hand-coded pure-pursuit driver. Writes runs/p1-gate/ and
reports/phase1-gate.html, and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import sys
import time

from sdc.phase1 import my_ppo, report
from sdc.phase1.train import pursuit_baseline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--out", default="runs/p1-gate")
    p.add_argument("--report", default="reports/phase1-gate.html")
    a = p.parse_args()

    t0 = time.perf_counter()
    pursuit_baseline(a.out + "-pursuit")
    m = my_ppo.train(a.seed, a.steps, a.out, label=f"my PPO, seed {a.seed}")
    elapsed = time.perf_counter() - t0
    c = m["compare"]
    comp, ratio = c["completion"], c["lap_time_ratio"]
    ok_comp = comp >= 0.95
    ok_time = ratio is not None and ratio <= 1.1
    verdict = "PASS" if ok_comp and ok_time else "FAIL"
    n_done = sum(r["finished"] for r in m["final_eval"]["tracks"])
    notes = [
        f"Gate verdict: {verdict}.",
        f"Held-out lap completion {n_done}/20 = {comp:.0%} (threshold 95%): {'pass' if ok_comp else 'fail'}.",
        f"Mean lap time {m['final_eval']['mean_lap_time']:.2f} s vs. pure pursuit {m['baseline']['mean_lap_time']:.2f} s "
        f"on tracks both finished: {ratio:.3f}× (threshold 1.1×): {'pass' if ok_time else 'fail'}."
        if ratio else "No held-out lap finished: fail.",
        f"Policy: the owner's PPO (sdc/phase1/my_ppo.py; pieces dictated by the owner, textbook clipping), "
        f"seed {a.seed}, {m['trained_timesteps']:,} steps, {m['train_wall_s']:.0f} s training on 1 CPU thread. "
        f"Whole command: {elapsed:.0f} s.",
    ]
    report.build([a.out + "-pursuit", a.out], "Phase 1 Gate", "Does the owner's own PPO finish at least 95% of laps on "
                 "20 unseen tracks within 1.1× the hand-coded driver's lap time?", a.report, eyebrow="Phase 1 gate",
                 notes=notes)
    print("\n".join(notes))
    print(f"Report: {a.report}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
