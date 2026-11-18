"""Phase 4.6 E9 report: python -m sdc.phase46.e9_report --notes "<html>" -> reports/phase46-e9.html

Reads runs/p46-e9-stitch.json (the stitched teacher), runs/p46-e9-s{0,1} (copying it) and runs/p46-e8-plain(-s1)
(copying the cautious teacher, same recipe); a map-view GIF of an E9 averaged car on a validation run where it got
stuck in a double.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch

from sdc.jumps import driver as dr
from sdc.jumps import loop as lp
from sdc.jumps import render_loop as rl
from sdc.phase4 import report as rp
from sdc.phase46 import common as cm
from sdc.phase46.bc_loop import student_env
from sdc.phase46.e3_report import endings
from sdc.phase46.ensemble import Ensemble

OUT = Path("reports/phase46-e9.html")
RUNS = (("p46-e8-plain", "cautious", 0), ("p46-e8-plain-s1", "cautious", 1), ("p46-e9-s0", "stitched", 0),
        ("p46-e9-s1", "stitched", 1))


def takeoff(eps):
    v = collections.defaultdict(list)
    for e in eps:
        for j in e["jumps"]:
            if "v" in j:
                v[j["kind"]].append(j["v"])
    return {k: sum(x) / len(x) for k, x in v.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="")
    ap.add_argument("--teacher-v", default='{"cautious": {"table": 5.39, "double": 5.79, "drop": 5.19}, '
                                           '"stitched": {"table": 5.44, "double": 6.06, "drop": 4.3}}')
    a = ap.parse_args()
    torch.set_num_threads(1)
    st = json.loads(Path("runs/p46-e9-stitch.json").read_text())["validation"]
    tv = json.loads(a.teacher_v)
    rows, vrows = [], []
    for name in ("cautious", "stitched"):
        t = st[name]
        rows.append([f"<i>{name} teacher</i>", "–", f"{t['finished']}/60", "–", endings(t), f"{t['lap_s']:.1f} s"])
        vrows.append([f"<i>{name} teacher</i>"] + [f"{tv[name][k]:.1f}" for k in ("table", "double", "drop")])
    for run, teacher, seed in RUNS:
        m = json.loads(Path(f"runs/{run}/metrics.json").read_text())
        f = m["final"]
        rows.append([f"students copying the {teacher} teacher", seed, f"<b>{f['finished_n']}/60</b>",
                     ", ".join(map(str, m["alone"])), endings(f), f"{f['lap_s']:.1f} s"])
        v = takeoff(m["final_episodes"])
        vrows.append([f"students, {teacher}, seed {seed}"] + [f"{v[k]:.1f}" for k in ("table", "double", "drop")])
    m = json.loads(Path("runs/p46-e9-s1/metrics.json").read_text())
    i = next(k for k, e in enumerate(m["final_episodes"]) if e["reason"] == "stuck")
    course, x0 = cm.VAL_SEEDS[i // len(dr.STARTS)], dr.STARTS[i % len(dr.STARTS)]
    env = {k: v for k, v in student_env(False, val=True).items() if k not in ("loop", "max_steps")}
    g = Path(f"runs/p46-e9-s1/loop-{course}.gif")
    ep = rl.gif(Ensemble([f"runs/p46-e9-s1/s{k}" for k in range(3)]), course, g,
                "E9 stitched-teacher students · average of 3", x0=x0, **env)
    fig = rp.figure(g, f"E9 seed 1, average of 3: validation loop {course}, start {x0:+.1f} m, gate-range physics, one "
                       f"run at real speed (in scoring it got stuck). Ended: {ep['reason']} after {ep['l'] * 0.02:.1f} s.")
    sections = [f"<h2>Result</h2><div class='note'>{a.notes}</div>",
                "<h2>Laps finished (validation loops, gate-range physics, 60 laps)</h2>" + rp.table(
                    ["driver", "seed", "average of 3", "each alone", "other endings", "median lap"], rows)
                + "<p class='caption'>Students: E5's three, 6 more rounds of 60k ordinary copying steps (E8's "
                  "control recipe), driven by their average. Only the teacher differs.</p>",
                "<h2>Takeoff speed by jump kind (m/s, mean over validation runs)</h2>" + rp.table(
                    ["driver", "table", "double", "drop"], vrows)
                + "<p class='caption'>Teachers: start 0 only (20 laps). Students: the averaged car, 60 laps.</p>",
                "<h2>A stuck double</h2><div class='gifs'>" + fig + "</div>"]
    rp.page("Phase 4.6 E9 teacher", "Phase 4.6 · E9 · validation loops 10000–10019", "A stitched teacher",
            "Copying a teacher with a jump speed per jump kind.", sections, OUT,
            footer="Code: sdc/phase46/e9_sweep.py, e9_stitch.py, hard.py (--jump-v), e9_report.py")
    print(OUT)


if __name__ == "__main__":
    main()
