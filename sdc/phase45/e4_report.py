"""Phase 4.5 E4 report: python -m sdc.phase45.e4_report --notes "<html>" --table JSON -> reports/phase45-e4.html"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sdc.jumps import render3d
from sdc.phase4 import report as rp
from sdc.phase4 import train as tr
from sdc.phase45.e1_report import WithEnv

OUT = Path("reports/phase45-e4.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="")
    ap.add_argument("--table", default="[]")
    a = ap.parse_args()
    torch.set_num_threads(1)
    rounds = []
    for name, run in (("baseline 64", "p45-steady-bc-s0"), ("(a) 256", "p45-e4-a-s0"), ("(c) + filter", "p45-e4-c-s0"),
                      ("(d) both", "p45-e4-d-s0")):
        m = json.loads(Path(f"runs/{run}/metrics.json").read_text())
        rounds.append([name] + [f"{r['mse']:.4f}" for r in m["rounds"]])
    pol, _ = tr.load(Path("runs/p45-e4-c-s0"))
    g = Path("runs/p45-e4-c-s0/e4-10000.gif")
    ep = render3d.gif(WithEnv(pol), 10000, g, "E4 (c) copy + your filter · gate range")
    sections = [f"<h2>Result</h2><div class='note'>{a.notes}</div>",
                "<h2>Landed (validation, 180 jumps)</h2>" + rp.table(
                    ["student", "seed 0", "seed 1", "training range (seed 0)", "imitation error"], json.loads(a.table))
                + "<p class='caption'>Copies of the steady expert (7.4 m/s), parts-list sensors, trained on training-range "
                  "physics. Seeds 0 and 1 scored at gate range. Imitation error: mean squared action error on the copy's own "
                  "training data after the last DAgger round.</p>",
                "<h2>Imitation error by DAgger round (seed 0)</h2>" + rp.table(
                    ["student"] + [f"round {i}" for i in range(6)], rounds)
                + "<p class='caption'>Round 0 is the expert's own driving; later rounds add states the student visits.</p>",
                "<h2>The car</h2><div class='gifs'>" + rp.figure(
                    g, f"3D chase view: (c) seed 0 on validation course 10000 at gate-range physics, one run, real speed. "
                       f"{ep['reason']}, landed {ep['landed']}/3.") + "</div>"]
    rp.page("Phase 4.5 E4 student needs", "Phase 4.5 · E4 · validation courses 10000–10019", "What does the student need?",
            "A bigger network, your filter's tilt as inputs, or both, for the copy of the steady expert.", sections, OUT,
            footer="Code: sdc/phase45/e4_runs.sh, e4_report.py")
    print(OUT)


if __name__ == "__main__":
    main()
