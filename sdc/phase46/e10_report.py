"""Phase 4.6 E10 report: python -m sdc.phase46.e10_report --notes "<html>" -> reports/phase46-e10.html

Reads runs/p46-e10-{a,b}-s{0,1}/metrics.json and runs/p46-e10-{a,b}-avg.json (bash sdc/phase46/e10_runs.sh); a
map-view GIF of each arm's averaged car on validation loop 10000 (truggy physics, gate range).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sdc.jumps import render_loop as rl
from sdc.phase4 import report as rp
from sdc.phase46.e3_report import endings
from sdc.phase46.ensemble import Ensemble

OUT = Path("reports/phase46-e10.html")
ARMS = (("a", "A: no expo"), ("b", "B: steering expo 0.4"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    rows = [["re-tuned teacher (truggy physics)", "–", "55/60", "rolled over 3, off the lane 2", "97%", "31.4 s"]]
    curves, figs = [], []
    for i, (arm, name) in enumerate(ARMS):
        runs = [f"runs/p46-e10-{arm}-s{s}" for s in (0, 1)]
        for s, run in enumerate(runs):
            m = json.loads((Path(run) / "metrics.json").read_text())
            f = m["final"]
            rows.append([name, f"seed {s}", f"{f['finished_n']}/60", endings(f), f"{f['rate']:.0%}", f"{f['lap_s']:.1f} s"])
            curves.append(dict(evals=[dict(timesteps=r["samples"], rate=r["finished"] / 20) for r in m["rounds"]],
                               _color=2 * i + s))
        f = json.loads(Path(f"runs/p46-e10-{arm}-avg.json").read_text())["final"]
        rows.append([f"<b>{name}</b>", "<b>average of 2</b>", f"<b>{f['finished_n']}/60</b>", endings(f),
                     f"{f['rate']:.0%}", f"{f['lap_s']:.1f} s"])
        env = {k: v for k, v in m["env46"].items() if k not in ("loop", "max_steps")}
        g = Path(f"runs/p46-e10-{arm}-s0/avg-10000.gif")
        ep = rl.gif(Ensemble(runs), 10000, g, f"E10 {name[:2]} average of 2 · truggy · gate range", **env)
        figs.append(rp.figure(g, f"{name}, average of 2: validation loop 10000, start 0, truggy physics, gate range, "
                                 f"one run at real speed. Ended: {ep['reason']} after {ep['l'] * 0.02:.1f} s."))
    sections = [f"<h2>Result</h2><div class='note'>{a.notes}</div>",
                "<h2>Laps finished (validation loops, gate-range physics, 60 laps)</h2>" + rp.table(
                    ["driver", "", "laps finished", "other endings", "jumps landed", "median lap"], rows)
                + "<p class='caption'>Your GRU + filter, parts-list sensors plus the 45° sensors, no jump notes. DAgger "
                  "copying of the re-tuned teacher on the truggy physics (ESC brake, limited-slip diffs, stronger motor, "
                  "heavier wheels, elliptic friction, sliding grip drop): 16 rounds × 60k steps. Timed against the "
                  "re-tuned teacher's laps.</p>",
                "<h2>During copying</h2>" + rp.curve(curves, y_label="validation laps finished (%)", threshold=80.0,
                                                     threshold_label="gate ≥ {:g}%")
                + rp.legend([(0, "A seed 0", ""), (1, "A seed 1", ""), (2, "B seed 0", ""), (3, "B seed 1", "")])
                + "<p class='caption'>After each round, start 0 only (20 laps: noisy).</p>",
                "<h2>The averaged cars on loop 10000</h2><div class='gifs'>" + "".join(figs) + "</div>"]
    rp.page("Phase 4.6 E10 truggy", "Phase 4.6 · E10 · validation loops 10000–10019", "Truggy physics, with and without expo",
            "Copying the re-tuned teacher on the corrected physics.", sections, OUT,
            footer="Code: sdc/phase46/bc_loop.py (--truggy --expo), e10_runs.sh, e10_report.py")
    print(OUT)


if __name__ == "__main__":
    main()
