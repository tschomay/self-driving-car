"""Phase 4.5 E3 report: python -m sdc.phase45.e3_report --notes "<html>" -> reports/phase45-e3.html"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sdc.jumps import render3d
from sdc.phase4 import report as rp
from sdc.phase4 import train as tr
from sdc.phase45.e1_report import WithEnv

OUT = Path("reports/phase45-e3.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="")
    ap.add_argument("--extra", default="", help="JSON: extra rows [[label, nominal, train, gate], ...]")
    a = ap.parse_args()
    torch.set_num_threads(1)
    rows, curves = [], []
    for k, arm in enumerate(("A", "B")):
        for s in (0, 1):
            bc = json.loads(Path(f"runs/p45-e3-{arm}-bc-s{s}/metrics.json").read_text())
            ft = json.loads(Path(f"runs/p45-e3-{arm}-s{s}/metrics.json").read_text())
            f = ft["final"]
            rows.append([f"{arm} ({'nominal' if arm == 'A' else 'training range'}), seed {s}", f"{bc['final']['rate']:.0%}",
                         f"{f['rate']:.0%} (step {ft['selection']['chosen'] // 1000}k)",
                         f"{ft['evals'][-1]['rate']:.0%}",
                         ", ".join(f"{r} {n}" for r, n in sorted(f["reasons"].items(), key=lambda kv: -kv[1]) if r != "finished"),
                         " · ".join(f"{kk} {v}" for kk, v in f["by_kind"].items())])
            curves.append(dict(evals=ft["evals"], _color=k))
    extra = json.loads(a.extra) if a.extra else []
    sections = [f"<h2>Result</h2><div class='note'>{a.notes}</div>",
                "<h2>Landed at gate-range physics</h2>" + rp.table(
                    ["run", "copy", "copy + PPO (kept)", "end of training", "other endings (of 60)", "by kind"], rows)
                + "<p class='caption'>Validation courses 10000–10019 × 3 starts (180 jumps), parts-list sensors, "
                  "gate-range physics (wider than B's training range), one car at a time. End of training: one start "
                  "per course (60 jumps).</p>",
                "<h2>Fine-tune learning curves</h2>" + rp.curve(curves)
                + rp.legend([(0, "A: trained at nominal physics", ""), (1, "B: trained on the training range", "")])
                + "<p class='caption'>Validation landing rate at gate-range physics, every 50k steps, one start per course "
                  "(60 jumps: noisy). Step 0 is the copy.</p>"]
    if extra:
        sections.append("<h2>The copies at each physics range</h2>" + rp.table(
            ["driver", "nominal", "training range", "gate range"], extra)
            + "<p class='caption'>180 validation jumps each, same scoring.</p>")
    pol, _ = tr.load(Path("runs/p45-e3-B-bc-s1"))
    p = Path("runs/p45-e3-B-bc-s1/e3-10000.gif")
    ep = render3d.gif(WithEnv(pol), 10000, p, "E3 copy (B, seed 1) · gate range")
    sections.append("<h2>The car</h2><div class='gifs'>" + rp.figure(
        p, f"3D chase view: the best copy (B, seed 1) on validation course 10000 at gate-range physics, one run, real "
           f"speed. {ep['reason']}, landed {ep['landed']}/3.") + "</div>")
    rp.page("Phase 4.5 E3 retrain", "Phase 4.5 · E3 · validation courses 10000–10019", "Retrain for the real car",
            "Copy the 4.5 expert on the parts-list sensors, then fine-tune with a finish bonus: nominal vs randomized "
            "training physics.", sections, OUT, footer="Code: sdc/phase45/e3_runs.sh, e3_report.py; sdc/phase4/bc.py, train.py")
    print(OUT)


if __name__ == "__main__":
    main()
