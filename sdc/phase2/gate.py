"""Phase 2 gate: the owner's CNN encoder, trained by behavior cloning, on the 20 held-out tracks.

Usage:
    python -m sdc.phase2.gate            # about 17 min on 1 CPU thread

Steps:
  1. The expert: Phase 1's SB3 ray policy (E1 seed 1). Trained here (about 90 s) if runs/p1-e1-s1 is missing.
  2. Behavior cloning with the owner's encoder (sdc/phase2/my_cnn.py): the E5 recipe, seed 0.
  3. Held-out laps (tracks 10000-10019) and saliency maps.

Gate (roadmap): held-out completion ≥ 90% of the Phase 1 policy's (Phase 1 gate: 20/20, so ≥ 18/20),
and the saliency shows the network keying on track edges, not artifacts. Measured as steering edge
enrichment > 1 (edge pixels matter more than an average pixel), plus the saliency figure for a visual check.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sdc.phase1 import train as p1_train
from sdc.phase2 import bc, report, saliency

PHASE1_COMPLETION = 1.0   # Phase 1 gate (owner's PPO, 2026-11-08): 20/20 held-out


def main(out: str = "runs/p2-gate"):
    t0 = time.perf_counter()
    out = Path(out)
    if not Path(bc.EXPERT).exists():
        p1_train.train(seed=1, total_timesteps=300_000, out=Path(bc.EXPERT).parent, label="PPO rays (Phase 1)")
    m = bc.train(seed=0, out=out, noise=0.3, encoder="mine", label="BC, owner's CNN (gate)")
    sal = saliency.analyze(out, out=Path("reports/p2-gate-saliency.png"))
    (out / "saliency.json").write_text(json.dumps(sal, indent=1))
    comp = m["final_eval"]["completion"]
    enr = sal["edge_enrichment"]["steer"]["median"]
    ok_comp = comp >= 0.9 * PHASE1_COMPLETION
    ok_edges = enr > 1.0
    verdict = dict(completion=comp, threshold=0.9 * PHASE1_COMPLETION, completion_pass=ok_comp,
                   lap_time_ratio=m["compare"]["lap_time_ratio"], steer_edge_enrichment=enr,
                   throttle_edge_enrichment=sal["edge_enrichment"]["throttle"]["median"], edges_pass=ok_edges,
                   gate=bool(ok_comp and ok_edges), wall_s=round(time.perf_counter() - t0, 1))
    (out / "gate.json").write_text(json.dumps(verdict, indent=1))
    report.build(["runs/p1-pursuit", str(out)] if Path("runs/p1-pursuit/metrics.json").exists() else [str(out)],
                 "Phase 2 gate", "Does the owner's CNN, copying the Phase 1 driver from camera images, finish ≥ 18 of "
                 "the 20 held-out tracks while watching the road edges?", "reports/p2-gate.html",
                 notes=[f"Held-out completion {comp:.0%} vs threshold ≥ 90% of Phase 1's 100%: "
                        f"{'pass' if ok_comp else 'FAIL'}.",
                        f"Steering edge enrichment {enr:.2f} (> 1 means edge pixels matter more than an average "
                        f"pixel): {'pass' if ok_edges else 'FAIL'}. Throttle: "
                        f"{verdict['throttle_edge_enrichment']:.2f}.",
                        f"Lap time {m['compare']['lap_time_ratio']:.3f}× pure pursuit on finished tracks.",
                        f"Whole command {verdict['wall_s'] / 60:.0f} min on 1 CPU thread.",
                        f"Verdict: {'PASS' if verdict['gate'] else 'FAIL'}."],
                 saliency=[("reports/p2-gate-saliency.png", "Owner's CNN (gate run), held-out track 10000.")],
                 saliency_caption="Saliency (SmoothGrad): how much a small change at each pixel would move the steering "
                                  "(middle) or throttle (bottom). Brighter orange = more influence.")
    print(json.dumps(verdict, indent=1))
    print(f"Phase 2 gate: {'PASS' if verdict['gate'] else 'FAIL'}")


if __name__ == "__main__":
    main()
