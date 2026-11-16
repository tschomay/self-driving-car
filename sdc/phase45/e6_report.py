"""Phase 4.5 E6 report: python -m sdc.phase45.e6_report --notes "<html>" -> reports/phase45-e6.html"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sdc.jumps import render3d
from sdc.phase4 import report as rp
from sdc.phase4 import train_rnn as trr
from sdc.phase45.e1_report import WithEnv

OUT = Path("reports/phase45-e6.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    rows = []
    for name, runs in (("E5: GRU + filter, no map", ["p45-e5-bc-s0", "p45-e5-bc-s1"]),
                       ("E6: + out-lap map", ["p45-e6-s0", "p45-e6-s1"])):
        for s, r in enumerate(runs):
            f = json.loads(Path(f"runs/{r}/metrics.json").read_text())["final"]
            rows.append([f"{name}, seed {s}", f"{f['rate']:.0%}", f"{f['finished']:.0%}",
                         " · ".join(f"{k} {v}" for k, v in f["by_kind"].items())])
    teach = [["steady expert", "84%"], ["steady expert + out-lap map (+0.6 m/s)", "88%"],
             ["per-jump expert (true design speeds)", "92%"]]
    pol, _ = trr.load(Path("runs/p45-e6-s1"))
    g = Path("runs/p45-e6-s1/e6-10000.gif")
    ep = render3d.gif(WithEnv(pol), 10000, g, "E6 GRU + map (seed 1) · rollers · gate range")
    sections = [f"<h2>Result</h2><div class='note'>{a.notes}</div>",
                "<h2>Gate range, roller courses (validation, 180 jumps)</h2>" + rp.table(
                    ["car", "landed", "courses finished", "by kind"], rows)
                + rp.table(["teacher", "landed"], teach)
                + "<p class='caption'>Copies (no fine-tuning), parts-list sensors, trained on training-range physics, scored "
                  "one car at a time from 3 starts. E6's maps come from a 2 m/s crawl of each course (pitch shape + forward "
                  "lidar scan); practice-course maps are cross-fitted.</p>",
                "<h2>The car</h2><div class='gifs'>" + rp.figure(
                    g, f"3D chase view: E6 seed 1 on validation course 10000 (rollers) at gate-range physics, one run, real "
                       f"speed. {ep['reason']}, landed {ep['landed']}/3.") + "</div>"]
    rp.page("Phase 4.5 E6 out-lap map", "Phase 4.5 · E6 · validation courses 10000–10019 (rollers)",
            "Does the out-lap map help?", "E5's car plus two inputs from a walk-through crawl of the course.",
            sections, OUT, footer="Code: sdc/phase45/outlap.py, maps.py, e6_runs.sh, e6_report.py")
    print(OUT)


if __name__ == "__main__":
    main()
