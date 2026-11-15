"""Phase 4.5 E5 report: python -m sdc.phase45.e5_report --notes "<html>" -> reports/phase45-e5.html"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sdc.jumps import render3d
from sdc.phase4 import report as rp
from sdc.phase4 import train_rnn as trr
from sdc.phase45.e1_report import WithEnv

OUT = Path("reports/phase45-e5.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    rows, curves = [], []
    for s in (0, 1):
        bc = json.loads(Path(f"runs/p45-e5-bc-s{s}/metrics.json").read_text())["final"]
        m = json.loads(Path(f"runs/p45-e5-s{s}/metrics.json").read_text())
        f = m["final"]
        for name, x in ((f"copy, seed {s}", bc), (f"copy + PPO (kept at {m['selection']['chosen'] // 1000}k), seed {s}", f)):
            rows.append([name, f"{x['rate']:.0%}", f"{x['finished']:.0%}",
                         ", ".join(f"{k} {v}" for k, v in sorted(x["reasons"].items(), key=lambda kv: -kv[1]) if k != "finished"),
                         " · ".join(f"{k} {v}" for k, v in x["by_kind"].items())])
        curves.append(dict(evals=m["evals"], _color=s))
    pol, _ = trr.load(Path("runs/p45-e5-bc-s0"))
    g = Path("runs/p45-e5-bc-s0/e5-10000.gif")
    ep = render3d.gif(WithEnv(pol), 10000, g, "E5 GRU copy (seed 0) · rollers · gate range")
    sections = [f"<h2>Result</h2><div class='note'>{a.notes}</div>",
                "<h2>Gate range, roller courses (validation, 180 jumps)</h2>" + rp.table(
                    ["car", "landed", "courses finished", "other endings (of 60)", "by kind"], rows)
                + "<p class='caption'>Your GRU + filter inputs, copying the steady expert, parts-list sensors, trained on "
                  "training-range physics; scored one car at a time from 3 starts on validation courses 10000–10019 "
                  "(roller versions). References: steady expert 84%, per-jump expert 92%.</p>",
                "<h2>Fine-tune learning curves</h2>" + rp.curve(curves)
                + rp.legend([(0, "seed 0", ""), (1, "seed 1", "")])
                + "<p class='caption'>Validation landing rate at gate range every 50k steps, one start per course (60 jumps: "
                  "noisy). Step 0 is the copy; the first 100k steps train only the critic.</p>",
                "<h2>The car</h2><div class='gifs'>" + rp.figure(
                    g, f"3D chase view: the seed 0 copy on validation course 10000 (rollers) at gate-range physics, one run, "
                       f"real speed. {ep['reason']}, landed {ep['landed']}/3.") + "</div>"]
    rp.page("Phase 4.5 E5 rollers", "Phase 4.5 · E5 · validation courses 10000–10019 (rollers)",
            "The best student on rollers", "Your GRU + filter car on the roller courses, then fine-tuned with your PPO + finish bonus.",
            sections, OUT, footer="Code: sdc/phase45/e5_runs.sh, bc_rnn.py, e5_report.py; sdc/phase4/train_rnn.py")
    print(OUT)


if __name__ == "__main__":
    main()
