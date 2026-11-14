"""Phase 4.5 E4(b) report: python -m sdc.phase45.e4b_report --notes "<html>" -> reports/phase45-e4b.html"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sdc.jumps import render3d
from sdc.phase4 import report as rp
from sdc.phase4 import train_rnn as trr
from sdc.phase45.e1_report import WithEnv

OUT = Path("reports/phase45-e4b.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    rows = []
    for name, runs in (("baseline: no memory, no filter (E3b)", ["p45-steady-bc-s0", "p45-steady-bc-s1"]),
                       ("(c) filter inputs, no memory", ["p45-e4-c-s0", "p45-e4-c-s1"]),
                       ("(b) your GRU + filter inputs", ["p45-e4-b-s0", "p45-e4-b-s1"])):
        ms = [json.loads(Path(f"runs/{r}/metrics.json").read_text())["final"] for r in runs]
        rows.append([name, " / ".join(f"{m['rate']:.0%}" for m in ms), " / ".join(f"{m['finished']:.0%}" for m in ms),
                     " / ".join(str(m["reasons"].get("off the lane", 0)) for m in ms),
                     " / ".join(str(m["reasons"].get("stuck", 0)) for m in ms),
                     " / ".join(str(m["reasons"].get("rolled over", 0)) for m in ms),
                     " · ".join(f"{k} {v}" for k, v in ms[0]["by_kind"].items())])
    pol, _ = trr.load(Path("runs/p45-e4-b-s0"))
    g = Path("runs/p45-e4-b-s0/e4b-10000.gif")
    ep = render3d.gif(WithEnv(pol), 10000, g, "E4(b) your GRU + filter · gate range")
    sections = [f"<h2>Result</h2><div class='note'>{a.notes}</div>",
                "<h2>Gate range (validation, 180 jumps per seed)</h2>" + rp.table(
                    ["student", "landed", "courses finished", "off the lane", "stuck", "rolled over", "by kind (seed 0)"], rows)
                + "<p class='caption'>Copies of the steady expert (7.4 m/s), parts-list sensors, trained on training-range "
                  "physics; seeds 0 / 1. Endings counted over 60 runs (20 courses × 3 starts).</p>",
                "<h2>The car</h2><div class='gifs'>" + rp.figure(
                    g, f"3D chase view: your GRU car (seed 0) on validation course 10000 at gate-range physics, one run, "
                       f"real speed. {ep['reason']}, landed {ep['landed']}/3.") + "</div>"]
    rp.page("Phase 4.5 E4b GRU memory", "Phase 4.5 · E4(b) · validation courses 10000–10019", "Does your GRU memory help?",
            "Your GRU with your filter's inputs, copying the steady expert, vs the same without memory.", sections, OUT,
            footer="Code: sdc/phase45/bc_rnn.py, e4b_runs.sh, e4b_report.py; sdc/phase4/my_gru.py (yours)")
    print(OUT)


if __name__ == "__main__":
    main()
