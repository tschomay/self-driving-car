"""Phase 0 gate: one command trains PPO on CartPole-v1 and checks the gate.

    python -m sdc.phase0.gate

Gate (docs/roadmap.md): held-out mean return >= 475 over 100 episodes, with the whole
command (env setup, training, evaluation, GIF, report) finishing in under 10 minutes.
Writes runs/gate/ and reports/phase0-gate.html, and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import sys
import time

from sdc import report
from sdc.phase0.train import SOLVED_RETURN, train

TIME_LIMIT_S = 600


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--out", default="runs/gate")
    p.add_argument("--report", default="reports/phase0-gate.html")
    a = p.parse_args()

    t0 = time.perf_counter()
    m = train("tuned", a.seed, a.steps, a.out, label=f"PPO tuned, seed {a.seed}")
    elapsed = time.perf_counter() - t0
    mean = m["final_eval"]["mean"]
    ok_return = mean >= SOLVED_RETURN
    ok_time = elapsed < TIME_LIMIT_S
    verdict = "PASS" if ok_return and ok_time else "FAIL"
    notes = [
        f"Gate verdict: {verdict}.",
        f"Held-out mean return {mean:.1f} (threshold {SOLVED_RETURN:g}): {'pass' if ok_return else 'fail'}.",
        f"Wall clock for train + 100-episode eval + GIF: {elapsed:.0f} s (limit {TIME_LIMIT_S} s): "
        f"{'pass' if ok_time else 'fail'}.",
    ]
    report.build([a.out], "Phase 0 Gate", "Can one command train PPO to solve CartPole-v1 in under 10 minutes?",
                 a.report, eyebrow="Phase 0 gate", notes=notes)
    total = time.perf_counter() - t0
    print("\n".join(notes))
    print(f"Report: {a.report} (total {total:.0f} s)")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
