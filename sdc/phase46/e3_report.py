"""Phase 4.6 E3/E4 report: python -m sdc.phase46.e3_report [--exp e4] --notes "<html>" -> reports/phase46-<exp>.html

Reads runs/p46-<exp>-{map,nomap}-s{0,1}/metrics.json (bash sdc/phase46/e3_runs.sh or e4_runs.sh) and draws map-view
GIFs of the better seed of each arm on validation loop 10000 at gate-range physics. With --exp e4, E3's numbers are
shown alongside.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sdc.jumps import render_loop as rl
from sdc.phase4 import report as rp
from sdc.phase46 import common as cm
from sdc.phase46.bc_loop import student_env

ARMS = (("nomap", "A: sensors only"), ("map", "B: sensors + out-lap map"))


def endings(sc):
    return ", ".join(f"{k} {v}" for k, v in sorted(sc["reasons"].items(), key=lambda kv: -kv[1]) if k != "finished") or "–"


class Policy:
    def __init__(self, run, env):
        self.pol, self.env_kw = cm.load_student(run), env
        self.reset = self.pol.reset

    def __call__(self, b):
        return self.pol(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="")
    ap.add_argument("--exp", default="e3")
    a = ap.parse_args()
    OUT = Path(f"reports/phase46-{a.exp}.html")
    torch.set_num_threads(1)
    rows, curves, best = [], [], {}
    maps = json.loads(Path("runs/p46-maps-val.json").read_text())
    for i, (arm, name) in enumerate(ARMS):
        for s in (0, 1):
            run = Path(f"runs/p46-{a.exp}-{arm}-s{s}")
            m = json.loads((run / "metrics.json").read_text())
            f = m["final"]
            eps = m["final_episodes"]
            mapped = [e for e in eps if maps[f"{e['course']}|{e.get('x0', 0.0):.2f}"]["kappa"]] if "x0" in eps[0] else eps
            far = np.median([min(e["x"] / 175.0, 1) for e in eps])
            prev = ""
            if a.exp != "e3":
                p3 = json.loads(Path(f"runs/p46-e3-{arm}-s{s}/metrics.json").read_text())["final"]["finished_n"]
                prev = f" (E3: {p3})"
            rows.append([name, s, f"<b>{f['finished_n']}/60</b>{prev}", f"{f['in_time_n']}/60", endings(f), f"{f['rate']:.0%}",
                         f"{f['lap_s']:.1f} s" if f["finished_n"] else "–", f"{m['rounds'][-1]['mse']:.4f}"])
            curves.append(dict(evals=[dict(timesteps=r["samples"], rate=r["finished"] / 20) for r in m["rounds"]],
                               _color=2 * i + s))
            if arm not in best or f["finished_n"] > best[arm][1]:
                best[arm] = (str(run), f["finished_n"], s)
    figs = []
    for arm, name in ARMS:
        run, _, s = best[arm]
        g = Path(run) / "loop-10000.gif"
        ep = rl.gif(Policy(run, {}), 10000, g, f"{a.exp.upper()} {name[:2]} seed {s} · gate range",
                    **{k: v for k, v in student_env(arm == "map", val=True).items() if k not in ("loop", "max_steps")})
        figs.append(rp.figure(g, f"{name}, seed {s}: validation loop 10000, start 0, gate-range physics, one run at real "
                                 f"speed. Ended: {ep['reason']} after {ep['l'] * 0.02:.1f} s, landed {ep['landed']}/3."))
    sections = [f"<h2>Result</h2><div class='note'>{a.notes}</div>",
                "<h2>Laps finished (validation loops, gate-range physics, 60 laps)</h2>" + rp.table(
                    ["student", "seed", "laps finished", "within 1.5× teacher's time", "other endings", "jumps landed",
                     "median lap", "imitation error"], rows)
                + "<p class='caption'>Your GRU + filter inputs, parts-list sensors plus two 45° range sensors, copied "
                  f"(DAgger, {len(m['rounds'])} rounds × 60k steps) on 291 practice loops at training-range physics; no fine-tuning. "
                  "Scored one car at a time on validation loops 10000–10019 × 3 starts, each with its own out-lap map "
                  "(6 of 60 out-laps failed: those runs race without a map). References: true-track teacher 56/60, "
                  "map-following teacher 51/60, Phase 4.5 car 0/60.</p>",
                "<h2>During copying</h2>" + rp.curve(curves, y_label="validation laps finished (%)",
                                                     threshold=80.0, threshold_label="gate ≥ {:g}%")
                + rp.legend([(0, "A seed 0", ""), (1, "A seed 1", ""), (2, "B seed 0", ""), (3, "B seed 1", "")])
                + "<p class='caption'>After each round, start 0 only (20 laps: noisy). x axis: training samples.</p>",
                "<h2>The students on loop 10000</h2><div class='gifs'>" + "".join(figs) + "</div>"]
    title = dict(e3="Copy the teacher on loops", e4="Train longer").get(a.exp, a.exp)
    rp.page(f"Phase 4.6 {a.exp.upper()} copies", f"Phase 4.6 · {a.exp.upper()} · validation loops 10000–10019", title,
            "Your GRU car copying the loop teacher, with and without the out-lap map of the turns.", sections, OUT,
            footer=f"Code: sdc/phase46/bc_loop.py, {a.exp}_runs.sh, e3_report.py, maps.py")
    print(OUT)


if __name__ == "__main__":
    main()
