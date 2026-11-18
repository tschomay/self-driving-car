"""Phase 4.6 E5 report: python -m sdc.phase46.e5_report --notes "<html>" -> reports/phase46-e5.html

Reads runs/p46-e5-s{0,1,2}/metrics.json (sensors-only students copying the teacher with margin) and E4's sensors-only
students for comparison; draws a map-view GIF of the best E5 seed on validation loop 10000 at gate-range physics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sdc.jumps import render_loop as rl
from sdc.phase4 import report as rp
from sdc.phase46.bc_loop import student_env
from sdc.phase46.e3_report import Policy, endings

OUT = Path("reports/phase46-e5.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    rows, curves, best = [], [], None
    for exp, seeds, label in (("e4-nomap", (0, 1), "E4: standard teacher"), ("e5", (0, 1, 2), "E5: teacher with margin")):
        for s in seeds:
            run = Path(f"runs/p46-{exp}-s{s}")
            m = json.loads((run / "metrics.json").read_text())
            f = m["final"]
            rows.append([label, s, f"<b>{f['finished_n']}/60</b>", f"{f['in_time_n']}/60", endings(f), f"{f['rate']:.0%}",
                         f"{f['lap_s']:.1f} s", f"{m['rounds'][-1]['mse']:.4f}"])
            if exp == "e5":
                curves.append(dict(evals=[dict(timesteps=r["samples"], rate=r["finished"] / 20) for r in m["rounds"]],
                                   _color=s))
                if best is None or f["finished_n"] > best[1]:
                    best = (str(run), f["finished_n"], s)
    run, _, s = best
    g = Path(run) / "loop-10000.gif"
    env = {k: v for k, v in student_env(False, val=True).items() if k not in ("loop", "max_steps")}
    ep = rl.gif(Policy(run, {}), 10000, g, f"E5 seed {s} · teacher with margin · gate range", **env)
    fig = rp.figure(g, f"E5 seed {s}: validation loop 10000, start 0, gate-range physics, one run at real speed. "
                       f"Ended: {ep['reason']} after {ep['l'] * 0.02:.1f} s, landed {ep['landed']}/3.")
    sections = [f"<h2>Result</h2><div class='note'>{a.notes}</div>",
                "<h2>Laps finished (validation loops, gate-range physics, 60 laps)</h2>" + rp.table(
                    ["student", "seed", "laps finished", "within 1.5× standard teacher's time", "other endings",
                     "jumps landed", "median lap", "imitation error"], rows)
                + "<p class='caption'>Sensors-only students (your GRU + filter, parts-list sensors plus two 45° range "
                  "sensors), copied with DAgger for 16 rounds × 60k steps. E5's teacher takes corners with a 2.5 m/s² "
                  "grip budget instead of 3.5 and brakes at 2.5 m/s² instead of 3.0; on its own it finishes 55/60 at "
                  "32.5 s (standard teacher 56/60, 30.1 s). Both are timed against the standard teacher.</p>",
                "<h2>During copying</h2>" + rp.curve(curves, y_label="validation laps finished (%)",
                                                     threshold=80.0, threshold_label="gate ≥ {:g}%")
                + rp.legend([(0, "seed 0", ""), (1, "seed 1", ""), (2, "seed 2", "")])
                + "<p class='caption'>After each round, start 0 only (20 laps: noisy). x axis: training samples.</p>",
                "<h2>The best E5 student on loop 10000</h2><div class='gifs'>" + fig + "</div>"]
    rp.page("Phase 4.6 E5 margin", "Phase 4.6 · E5 · validation loops 10000–10019", "Teacher with margin",
            "Your GRU car copying a teacher that leaves grip in reserve through the turns.", sections, OUT,
            footer="Code: sdc/phase46/bc_loop.py (--a-lat 2.5 --a-brake 2.5), e5_runs.sh, e5_report.py")
    print(OUT)


if __name__ == "__main__":
    main()
