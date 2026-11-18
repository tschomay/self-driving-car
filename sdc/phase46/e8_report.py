"""Phase 4.6 E8 report: python -m sdc.phase46.e8_report --notes "<html>" -> reports/phase46-e8.html

Reads runs/p46-e8-{hard,plain}/metrics.json (sdc/phase46/hard.py) and draws a map-view GIF of each arm's averaged
car on validation loop 10000 at gate-range physics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sdc.jumps import render_loop as rl
from sdc.phase4 import report as rp
from sdc.phase46.bc_loop import student_env
from sdc.phase46.e3_report import endings
from sdc.phase46.ensemble import Ensemble

OUT = Path("reports/phase46-e8.html")
ARMS = (("hard", "hard spots"), ("plain", "control: ordinary laps"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    rows = [["E7: E5 students, no more training", "50/60", "43, 47, 49", "off the lane 7, rolled over 3", "32.8 s", "–"]]
    curves, dis, figs = [], [], []
    env = {k: v for k, v in student_env(False, val=True).items() if k not in ("loop", "max_steps")}
    for i, (arm, name) in enumerate(ARMS):
        run = Path(f"runs/p46-e8-{arm}")
        m = json.loads((run / "metrics.json").read_text())
        f = m["final"]
        rows.append([f"E8 {name}", f"<b>{f['finished_n']}/60</b>", ", ".join(map(str, m["alone"])), endings(f),
                     f"{f['lap_s']:.1f} s", f"{sum(r['branch_steps'] for r in m['rounds']):,} "
                                            f"({sum(r['hard_spots'] for r in m['rounds'])} spots)"])
        steps = 0
        ev, dv = [], []
        for r in m["rounds"]:
            steps += r["main_steps"] + r["branch_steps"]
            ev.append(dict(timesteps=steps, rate=r["finished"] / 20))
            dv.append(dict(timesteps=steps, rate=r["disagree_mean"]))
        curves.append(dict(evals=ev, _color=i))
        dis.append(dict(evals=dv, _color=i))
        g = run / "loop-10000.gif"
        ep = rl.gif(Ensemble([str(run / f"s{k}") for k in range(3)]), 10000, g, f"E8 {name} · average of 3 · gate range",
                    **env)
        figs.append(rp.figure(g, f"E8 {name}, average of 3: validation loop 10000, start 0, gate-range physics, one run "
                                 f"at real speed. Ended: {ep['reason']} after {ep['l'] * 0.02:.1f} s."))
    sections = [f"<h2>Result</h2><div class='note'>{a.notes}</div>",
                "<h2>Laps finished (validation loops, gate-range physics, 60 laps)</h2>" + rp.table(
                    ["car", "average of 3", "each alone", "other endings (average)", "median lap", "replayed steps"],
                    rows)
                + "<p class='caption'>Both arms start from E5's three students and copy the cautious teacher for 6 "
                  "more rounds of 60k steps, driven by the three students' average. Hard spots: up to half of each "
                  "round re-drives 2.6 s from 1.2 s before a moment the students disagreed (4 times, extra noise). "
                  "Round 1 is ordinary driving in both arms.</p>",
                "<h2>During training</h2>" + rp.curve(curves, y_label="validation laps finished (%)", threshold=80.0,
                                                      threshold_label="gate ≥ {:g}%")
                + rp.legend([(0, "hard spots", ""), (1, "control", "")])
                + "<p class='caption'>The average of 3 after each round, start 0 only (20 laps: noisy).</p>"
                + rp.curve(dis, y_label="mean disagreement (std of actions)")
                + rp.legend([(0, "hard spots", ""), (1, "control", "")])
                + "<p class='caption'>How much the three students disagree while driving ordinary laps, per round.</p>",
                "<h2>The averaged cars on loop 10000</h2><div class='gifs'>" + "".join(figs) + "</div>"]
    rp.page("Phase 4.6 E8 hard spots", "Phase 4.6 · E8 · validation loops 10000–10019", "Copy more where they disagree",
            "Replaying the moments the students disagree, against the same amount of ordinary copying.", sections, OUT,
            footer="Code: sdc/phase46/hard.py, e8_report.py")
    print(OUT)


if __name__ == "__main__":
    main()
