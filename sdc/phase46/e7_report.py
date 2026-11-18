"""Phase 4.6 E7 report: python -m sdc.phase46.e7_report --notes "<html>" -> reports/phase46-e7.html

Reads runs/p46-e7.json (the average of E5's three students, sdc/phase46/ensemble.py) and the E5 students, compares
them run by run (by how many students finished alone), and draws a map-view GIF of the average on a run it rescued.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sdc.jumps import driver as dr
from sdc.jumps import render_loop as rl
from sdc.phase4 import report as rp
from sdc.phase46 import common as cm
from sdc.phase46.bc_loop import student_env
from sdc.phase46.e3_report import Policy, endings
from sdc.phase46.ensemble import Ensemble

OUT = Path("reports/phase46-e7.html")
RUNS = [f"runs/p46-e5-s{s}" for s in (0, 1, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    e7 = json.loads(Path("runs/p46-e7.json").read_text())
    ms = [json.loads((Path(r) / "metrics.json").read_text()) for r in RUNS]
    rows = [["E5 student alone", s, f"{m['final']['finished_n']}/60", endings(m["final"]), f"{m['final']['lap_s']:.1f} s"]
            for s, m in enumerate(ms)]
    f = e7["final"]
    rows.append(["<b>average of the three</b>", "0+1+2", f"<b>{f['finished_n']}/60</b>", endings(f), f"{f['lap_s']:.1f} s"])
    av = [e["finished"] for e in e7["final_episodes"]]
    alone = [sum(m["final_episodes"][i]["finished"] for m in ms) for i in range(60)]
    split = [[k, sum(n == k for n in alone), sum(av[i] for i in range(60) if alone[i] == k)] for k in (3, 2, 1, 0)]
    env = {k: v for k, v in student_env(False, val=True).items() if k not in ("loop", "max_steps")}
    pick = [i for i in range(60) if av[i] and alone[i] <= 1]
    i = pick[0]
    course, x0 = cm.VAL_SEEDS[i // len(dr.STARTS)], dr.STARTS[i % len(dr.STARTS)]
    best = max(range(3), key=lambda s: ms[s]["final"]["finished_n"])
    figs = []
    for who, pol, tag in (("average of 3", Ensemble(RUNS), "avg"), (f"seed {best} alone", Policy(RUNS[best], {}), f"s{best}")):
        g = Path(f"runs/p46-e7-{tag}-{course}.gif")
        ep = rl.gif(pol, course, g, f"E7 {who} · gate range", x0=x0, **env)
        figs.append(rp.figure(g, f"{who}: validation loop {course}, start {x0:+.1f} m, gate-range physics, one run at "
                                 f"real speed (in scoring, the average finished; {alone[i]} of 3 students alone did). "
                                 f"Ended: {ep['reason']} after {ep['l'] * 0.02:.1f} s."))
    sections = [f"<h2>Result</h2><div class='note'>{a.notes}</div>",
                "<h2>Laps finished (validation loops, gate-range physics, 60 laps)</h2>" + rp.table(
                    ["driver", "seed", "laps finished", "other endings", "median lap"], rows)
                + "<p class='caption'>The averaged car runs E5's three students (copies of the teacher with margin) "
                  "side by side, each with its own memory, and drives with the mean of their throttle and steering. "
                  "No new training. Same 60 runs as E5. Gate: 48/60 (on the test loops).</p>",
                "<h2>Run by run</h2>" + rp.table(["students that finish alone", "runs", "average finishes"], split)
                + "<p class='caption'>Each of the 60 runs grouped by how many of the three students finished it alone.</p>",
                "<h2>A run the average rescued</h2><div class='gifs'>" + "".join(figs) + "</div>"
                + "<p class='caption'>GIFs are re-driven, so physics noise can make one end differently from scoring.</p>"]
    rp.page("Phase 4.6 E7 average", "Phase 4.6 · E7 · validation loops 10000–10019", "Average the cautious students",
            "Three copies of the cautious teacher, driving one car together.", sections, OUT,
            footer="Code: sdc/phase46/ensemble.py, e7_report.py")
    print(OUT)


if __name__ == "__main__":
    main()
