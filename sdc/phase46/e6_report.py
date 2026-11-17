"""Phase 4.6 E6 report: python -m sdc.phase46.e6_report --notes "<html>" -> reports/phase46-e6.html

Reads runs/p46-e6-{nomap,map}.json (sdc/phase46/ensemble.py) and the E4 students they average, compares them run by
run, and draws map-view GIFs of one run per arm where the averaged car finished but both students alone did not.
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

ARMS = (("nomap", "A: sensors only"), ("map", "B: sensors + out-lap map"))
OUT = Path("reports/phase46-e6.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    rows, split, figs = [], [], []
    for arm, name in ARMS:
        e6 = json.loads(Path(f"runs/p46-e6-{arm}.json").read_text())
        runs = [f"runs/p46-e4-{arm}-s{s}" for s in (0, 1)]
        ms = [json.loads((Path(r) / "metrics.json").read_text()) for r in runs]
        for s, m in enumerate(ms):
            f = m["final"]
            rows.append([name, f"E4 seed {s} alone", f"{f['finished_n']}/60", endings(f), f"{f['lap_s']:.1f} s"])
        f = e6["final"]
        rows.append([name, "<b>average of the two</b>", f"<b>{f['finished_n']}/60</b>", endings(f), f"{f['lap_s']:.1f} s"])
        av = [e["finished"] for e in e6["final_episodes"]]
        s0, s1 = ([e["finished"] for e in m["final_episodes"]] for m in ms)
        both_fail = [i for i in range(60) if av[i] and not s0[i] and not s1[i]]
        split.append([name, sum(x and y for x, y in zip(s0, s1)), sum(x != y for x, y in zip(s0, s1)),
                      sum(not x and not y for x, y in zip(s0, s1)),
                      f"{sum(av[i] for i in range(60) if s0[i] and s1[i])} / {sum(s0[i] and s1[i] for i in range(60))}",
                      f"{sum(av[i] for i in range(60) if s0[i] != s1[i])} / {sum(s0[i] != s1[i] for i in range(60))}",
                      f"{len(both_fail)} / {sum(not x and not y for x, y in zip(s0, s1))}"])
        env = {k: v for k, v in student_env(arm == "map", val=True).items() if k not in ("loop", "max_steps")}
        i = both_fail[0]
        course, x0 = cm.VAL_SEEDS[i // len(dr.STARTS)], dr.STARTS[i % len(dr.STARTS)]
        for who, pol, tag in (("average", Ensemble(runs), "avg"), ("seed 0 alone", Policy(runs[0], {}), "s0")):
            g = Path(f"runs/p46-e6-{arm}-{tag}-{course}.gif")
            ep = rl.gif(pol, course, g, f"E6 {name[:2]} {who} · gate range", x0=x0, **env)
            figs.append(rp.figure(g, f"{name}, {who}: validation loop {course}, start {x0:+.1f} m, gate-range physics, "
                                     f"one run at real speed (a run where, in scoring, the average finished and both "
                                     f"students alone failed). Ended: {ep['reason']} after {ep['l'] * 0.02:.1f} s."))
    sections = [f"<h2>Result</h2><div class='note'>{a.notes}</div>",
                "<h2>Laps finished (validation loops, gate-range physics, 60 laps)</h2>" + rp.table(
                    ["arm", "driver", "laps finished", "other endings", "median lap"], rows)
                + "<p class='caption'>The averaged car runs both E4 students side by side, each with its own memory, "
                  "and drives with the mean of their throttle and steering at every step. No new training. Same 60 runs "
                  "as E4. True-track teacher: 56/60.</p>",
                "<h2>Run by run</h2>" + rp.table(
                    ["arm", "both students finish", "one finishes", "neither", "average finishes when both did",
                     "…when one did", "…when neither did"], split)
                + "<p class='caption'>Each of the 60 runs sorted by how the two students did alone, then how often "
                  "the averaged car finished it.</p>",
                "<h2>A run the average rescued</h2><div class='gifs'>" + "".join(figs) + "</div>"
                + "<p class='caption'>GIFs are re-driven, so physics noise can make one end differently from scoring.</p>"]
    rp.page("Phase 4.6 E6 averaging", "Phase 4.6 · E6 · validation loops 10000–10019", "Average the students",
            "Two copies of the same lesson, driving one car together.", sections, OUT,
            footer="Code: sdc/phase46/ensemble.py, e6_report.py")
    print(OUT)


if __name__ == "__main__":
    main()
