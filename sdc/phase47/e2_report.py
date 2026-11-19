"""Phase 4.7 E2 report: python -m sdc.phase47.e2_report --notes "<html>" -> reports/phase47-e2.html

Reads runs/p47-e2-prep.json (base car's exam), runs/p47-e2-s{0,1}/metrics.json (bash sdc/phase47/e2_runs.sh) and
runs/p47-e2-check.json (sdc.phase47.check); map-view GIFs of the base car and seed 0's kept car on validation loop 10000.
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
from sdc.phase47 import drills as dl
from sdc.phase47 import drills2 as d2
from sdc.phase47.train_residual import BASE, env_kw

OUT = Path("reports/phase47-e2.html")
SEEDS = (0, 1)
SEL_RUNS = 150          # 50 selection loops x 3 starts
TEACHER_SEL = 141       # the teacher on the same 150 runs (e2_prep)


def pct(x):
    return f"{x:.0%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    prep = json.loads(Path("runs/p47-e2-prep.json").read_text())
    ms = [json.loads(Path(f"runs/p47-e2-s{s}/metrics.json").read_text()) for s in SEEDS]
    be = prep["base_exam"]

    exam_rows = [["base car (step 0)", ""] + [pct(be[k]) for k in ("all",) + d2.KINDS]]
    for s, m in zip(SEEDS, ms):
        ex = {e["steps"]: e for e in m["exams"]}
        best = min(m["exams"], key=lambda e: (e["all"], e["steps"]))
        kept = ex.get(m["chosen"]["steps"])
        for label, e in (("best exam", best), ("kept checkpoint", kept), ("end (1M)", m["exams"][-1])):
            if e is None:
                continue
            exam_rows.append([f"seed {s}, {label}", f"{e['steps'] // 1000}k"] + [pct(e[k]) for k in ("all",) + d2.KINDS])
    exam_curves = [dict(evals=[dict(timesteps=e["steps"], rate=e["all"]) for e in m["exams"]], _color=s)
                   for s, m in zip(SEEDS, ms)]
    kind_curves = []
    for j, k in enumerate(d2.KINDS):
        both = [[e[k] for e in m["exams"]] for m in ms]
        kind_curves.append(dict(evals=[dict(timesteps=e["steps"], rate=(both[0][i] + both[1][i]) / 2)
                                       for i, e in enumerate(ms[0]["exams"])], _color=j))

    sel_rows = [["teacher (re-tuned, truggy)", "", f"{TEACHER_SEL}/{SEL_RUNS}", "", "", ""]]
    for s, m in zip(SEEDS, ms):
        for e in m["evals"]:
            mark = " (kept)" if e["steps"] == m["chosen"]["steps"] else ""
            sel_rows.append([f"seed {s}" + mark, f"{e['steps'] // 1000}k", f"{e['in_time']}/{SEL_RUNS}", endings(e),
                             pct(e["landed"]), f"{e['lap_s']:.1f} s"])
    sel_curves = [dict(evals=[dict(timesteps=e["steps"], rate=e["in_time"] / SEL_RUNS) for e in m["evals"]], _color=s)
                  for s, m in zip(SEEDS, ms)]

    ck = json.loads(Path("runs/p47-e2-check.json").read_text())
    names = {"teacher": "teacher", "base": "base car"}
    for s, m in zip(SEEDS, ms):
        names[f"runs/p47-e2-s{s}/model.pt"] = f"seed {s}, kept ({m['chosen']['steps'] // 1000}k)"
        names[f"runs/p47-e2-s{s}/ckpt/1001472.pt"] = f"seed {s}, end (1M)"
    check_rows = [[names.get(k, k), f"{v['in_time']}/{v['runs']}", endings(v), pct(v["landed"]), f"{v['lap_s']:.1f} s"]
                  for k, v in ck.items()]

    env = {k: v for k, v in env_kw(val=True).items() if k not in ("loop", "max_steps")}
    od = ms[0]["obs_dim"]
    g = Path("runs/p47-e2-s0/chosen-10000.gif")
    ep = rl.gif(dl.ResidualPolicy(BASE, torch.load("runs/p47-e2-s0/model.pt"), od), 10000, g,
                "E2 seed 0 · base + residual · truggy · gate range", **env)
    g2 = Path("runs/p47-e1-s0/base-10000.gif")
    if not g2.exists():
        rl.gif(Ensemble(BASE), 10000, g2, "base car · truggy · gate range", **env)
    figs = [rp.figure(g2, "Base car alone: validation loop 10000, start 0, one run at real speed."),
            rp.figure(g, f"Base + seed 0's kept E2 residual: the same run. Ended: {ep['reason']} after "
                         f"{ep['l'] * 0.02:.1f} s.")]

    amps = prep["amps"]
    sections = [
        f"<h2>Result</h2><div class='note'>{a.notes}</div>",
        "<h2>Drill exam (300 fixed drills, 60 per kind, never trained on)</h2>"
        + rp.table(["car", "steps", "all", *d2.KINDS], exam_rows)
        + "<p class='caption'>Share of exam drills that ended in a crash (lower is better). Exam drills come from 30 "
          "loops kept out of training, with fixed seeds, so every checkpoint drives the same 300. Strengths: "
          + ", ".join(f"{k} {amps[k]:.2f}" for k in d2.TUNED)
          + " (calibrated so the base car crashes about 40% of each); crash starts are not calibrated.</p>",
        "<h2>Exam crash rate during training</h2>"
        + rp.curve(exam_curves, y_label="exam drills crashed (%)", threshold=100 * be["all"],
                   threshold_label="base car {:.0f}%")
        + rp.legend([(0, "seed 0", ""), (1, "seed 1", "")])
        + "<h3>By drill kind (average of the two seeds)</h3>"
        + rp.curve(kind_curves, y_label="exam drills crashed (%)", threshold=100 * be["all"],
                   threshold_label="base car, all {:.0f}%")
        + rp.legend([(j, k, f"base {pct(be[k])}") for j, k in enumerate(d2.KINDS)]),
        f"<h2>Selection laps ({SEL_RUNS} runs: 50 practice loops held out × 3 starts)</h2>"
        + rp.curve(sel_curves, y_label="laps finished in time (%)", threshold=100 * TEACHER_SEL / SEL_RUNS,
                   threshold_label="teacher {:.0f}%")
        + rp.legend([(0, "seed 0", ""), (1, "seed 1", "")])
        + rp.table(["car", "steps", "laps in time", "other endings", "jumps landed", "median lap"], sel_rows)
        + "<p class='caption'>The kept checkpoint is the best of six on these runs (step 0, the base car, wins ties).</p>",
        "<h2>Fresh-loop check (loops 47000–47033 × 3 starts, never used before)</h2>"
        + rp.table(["driver", "laps finished", "other endings", "jumps landed", "median lap"], check_rows)
        + "<p class='caption'>Same 102 runs for every car, gate-range physics; laps timed against the teacher on the "
          "same run. The teacher (E1's check, same runs) finished 95/102 with a 31.2 s median lap.</p>",
        "<h2>Loop 10000, before and after</h2><div class='gifs'>" + "".join(figs) + "</div>"]
    rp.page("Phase 4.7 E2 drills", "Phase 4.7 · E2 · calibrated drills and a drill exam", "Harder drills",
            "With drills hard enough to crash the base car 40% of the time, does the residual learn to crash less, "
            "and does that carry over to full laps?", sections, OUT,
            footer="Code: sdc/phase47/drills2.py, e2_prep.py, train_residual.py, check.py, e2_runs.sh, e2_report.py")
    print(OUT)


if __name__ == "__main__":
    main()
