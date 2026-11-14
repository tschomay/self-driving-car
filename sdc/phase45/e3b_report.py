"""Phase 4.5 E3b report: python -m sdc.phase45.e3b_report --notes "<html>" -> reports/phase45-e3b.html"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sdc.jumps import render3d
from sdc.phase4 import report as rp
from sdc.phase4 import train as tr
from sdc.phase45.e1_report import WithEnv

OUT = Path("reports/phase45-e3b.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="")
    ap.add_argument("--table", default="[]", help="JSON rows [[driver, nominal, training range, gate range], ...]")
    a = ap.parse_args()
    torch.set_num_threads(1)
    curves = [dict(evals=json.loads(Path(f"runs/{p}-s{s}/metrics.json").read_text())["evals"], _color=k)
              for k, p in enumerate(("p45-e3-B", "p45-e3b")) for s in (0, 1)]
    pol, _ = tr.load(Path("runs/p45-steady-bc-s0"))
    g = Path("runs/p45-steady-bc-s0/e3b-10000.gif")
    ep = render3d.gif(WithEnv(pol), 10000, g, "E3b steady copy (seed 0) · gate range")
    sections = [f"<h2>Result</h2><div class='note'>{a.notes}</div>",
                "<h2>Landed (validation, 180 jumps each)</h2>" + rp.table(
                    ["driver", "nominal", "training range", "gate range"], json.loads(a.table))
                + "<p class='caption'>20 validation courses × 3 starts, parts-list sensors, one car at a time. "
                  "'–' = not measured.</p>",
                "<h2>Fine-tune learning curves (gate range)</h2>" + rp.curve(curves)
                + rp.legend([(0, "E3 B: copy of the per-jump-speed expert", ""), (1, "E3b: copy of the steady expert", "")])
                + "<p class='caption'>Validation landing rate at gate-range physics, every 50k steps, one start per course "
                  "(60 jumps: noisy). Step 0 is the copy.</p>",
                "<h2>The car</h2><div class='gifs'>" + rp.figure(
                    g, f"3D chase view: the steady copy (seed 0) on validation course 10000 at gate-range physics, one run, "
                       f"real speed. {ep['reason']}, landed {ep['landed']}/3.") + "</div>"]
    rp.page("Phase 4.5 E3b steady teacher", "Phase 4.5 · E3b · validation courses 10000–10019",
            "Copy a teacher the student can follow", "E3's recipe, but the expert drives one steady speed into every jump.",
            sections, OUT, footer="Code: sdc/phase45/e3b_runs.sh, e3b_report.py")
    print(OUT)


if __name__ == "__main__":
    main()
