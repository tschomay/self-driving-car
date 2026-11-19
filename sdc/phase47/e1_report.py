"""Phase 4.7 E1 report: python -m sdc.phase47.e1_report --notes "<html>" -> reports/phase47-e1.html

Reads runs/p47-e1-s{0,1}/metrics.json (bash sdc/phase47/e1_runs.sh) and runs/p47-e1-check.json (sdc.phase47.check);
a map-view GIF of seed 0's chosen car on validation loop 10000 (truggy physics, gate range).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sdc.jumps import render_loop as rl
from sdc.phase4 import report as rp
from sdc.phase46.e3_report import endings
from sdc.phase46.ensemble import Ensemble
from sdc.phase47 import drills as dl
from sdc.phase47.train_residual import BASE, env_kw

OUT = Path("reports/phase47-e1.html")
SEEDS = (0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    ms = [json.loads(Path(f"runs/p47-e1-s{s}/metrics.json").read_text()) for s in SEEDS]
    base = ms[0]["evals"][0]
    rows = [["teacher (re-tuned, truggy)", "", "55/60", "rolled over 3, off the lane 2", "97%", "31.4 s"],
            ["base car (frozen E10 pair)", "step 0", f"{base['in_time']}/60", endings(base), f"{base['landed']:.0%}",
             f"{base['lap_s']:.1f} s"]]
    for s, m in zip(SEEDS, ms):
        c = m["chosen"]
        rows.append([f"<b>base + residual, seed {s}</b>", f"kept: {c['steps'] // 1000}k", f"<b>{c['in_time']}/60</b>",
                     endings(c), f"{c['landed']:.0%}", f"{c['lap_s']:.1f} s"])
        e = m["evals"][-1]
        rows.append([f"base + residual, seed {s}", "end (1M)", f"{e['in_time']}/60", endings(e), f"{e['landed']:.0%}",
                     f"{e['lap_s']:.1f} s"])
    curves = [dict(evals=[dict(timesteps=e["steps"], rate=e["in_time"] / 60) for e in m["evals"]], _color=s)
              for s, m in zip(SEEDS, ms)]
    drill_rows = []
    for s, m in zip(SEEDS, ms):
        d = m["drills"]
        for kind in dl.KINDS:
            r = [x for x in d if x["kind"] == kind]
            if not r:
                continue
            q = len(r) // 4
            first, last = r[:q], r[-q:]
            cr = lambda xs: np.mean([x["reason"] in dl.CRASHES for x in xs])   # noqa: E731
            drill_rows.append([f"seed {s}", kind, f"{len(r)}", f"{cr(first):.0%}", f"{cr(last):.0%}"])
    ck = json.loads(Path("runs/p47-e1-check.json").read_text())
    names = {"teacher": "teacher", "base": "base car",
             "runs/p47-e1-s0/model.pt": "seed 0, kept checkpoint", "runs/p47-e1-s1/model.pt": "seed 1, kept checkpoint",
             "runs/p47-e1-s0/ckpt/1001472.pt": "seed 0, end (1M)", "runs/p47-e1-s1/ckpt/1001472.pt": "seed 1, end (1M)"}
    check_rows = [[names.get(k, k), f"{v['in_time']}/{v['runs']}", endings(v), f"{v['landed']:.0%}",
                   f"{v['lap_s']:.1f} s"] for k, v in ck.items()]
    env = {k: v for k, v in env_kw(val=True).items() if k not in ("loop", "max_steps")}
    od = ms[0]["obs_dim"]
    g = Path("runs/p47-e1-s0/chosen-10000.gif")
    ep = rl.gif(dl.ResidualPolicy(BASE, torch.load("runs/p47-e1-s0/model.pt"), od), 10000, g,
                "E1 seed 0 · base + residual · truggy · gate range", **env)
    g2 = Path("runs/p47-e1-s0/base-10000.gif")
    ep2 = rl.gif(Ensemble(BASE), 10000, g2, "base car · truggy · gate range", **env)
    figs = [rp.figure(g2, f"Base car alone: validation loop 10000, start 0, one run at real speed. Ended: {ep2['reason']} "
                          f"after {ep2['l'] * 0.02:.1f} s."),
            rp.figure(g, f"Base + seed 0's kept residual: the same run. Ended: {ep['reason']} after "
                         f"{ep['l'] * 0.02:.1f} s.")]
    sections = [f"<h2>Result</h2><div class='note'>{a.notes}</div>",
                "<h2>Validation laps (60 runs, gate-range physics)</h2>" + rp.table(
                    ["driver", "", "laps finished", "other endings", "jumps landed", "median lap"], rows)
                + "<p class='caption'>The kept checkpoint is the best of six validation scores (0, 200k … 1M steps); "
                  "step 0 is the base car exactly (zero correction). Picking the best of six flatters it: see the "
                  "fresh-loop check.</p>",
                "<h2>During training</h2>" + rp.curve(curves, y_label="validation laps finished (%)", threshold=80.0,
                                                      threshold_label="4.6 gate ≥ {:g}%")
                + rp.legend([(0, "seed 0", ""), (1, "seed 1", "")])
                + "<p class='caption'>60 validation runs every 200k steps. The base car is 52/60 = 87%; the teacher "
                  "55/60 = 92%.</p>",
                "<h2>Crash rate in drills, first vs last quarter of training</h2>" + rp.table(
                    ["seed", "drill", "drills driven", "crashed, first ¼", "crashed, last ¼"], drill_rows)
                + "<p class='caption'>With exploration noise on (it shrinks during training), so not the greedy car's "
                  "rate.</p>",
                "<h2>Fresh-loop check (loops 47000–47033 × 3 starts, never used before)</h2>" + rp.table(
                    ["driver", "laps finished", "other endings", "jumps landed", "median lap"], check_rows)
                + "<p class='caption'>Same 102 runs for every car, gate-range physics; laps timed against the teacher "
                  "on the same run. Nothing was trained, chosen or tuned on these loops.</p>",
                "<h2>Loop 10000, before and after</h2><div class='gifs'>" + "".join(figs) + "</div>"]
    rp.page("Phase 4.7 E1 drills", "Phase 4.7 · E1 · validation loops 10000–10019", "Drills alone",
            "Does a residual trained only in drills make the frozen copy finish more laps?", sections, OUT,
            footer="Code: sdc/phase47/drills.py, train_residual.py, check.py, e1_runs.sh, e1_report.py")
    print(OUT)


if __name__ == "__main__":
    main()
