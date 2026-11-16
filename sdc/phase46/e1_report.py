"""Phase 4.6 E1 report: python -m sdc.phase46.e1_report --notes "<html>" -> reports/phase46-e1.html

Reads runs/p46-e1/e1.json (python -m sdc.phase46.e1) and draws three GIFs on validation loop 10000 at gate-range
physics: the Phase 4.5 car (map and a 3D clip of its last seconds) and the teacher (map).
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
from sdc.phase46.e1 import load_car
from sdc.phase46.teacher import LoopExpert

OUT = Path("reports/phase46-e1.html")


def endings(sc):
    return ", ".join(f"{k} {v}" for k, v in sorted(sc["reasons"].items(), key=lambda kv: -kv[1]) if k != "finished") or "–"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/p46-e1/e1.json")
    ap.add_argument("--run", default="runs/p45-e5-bc-s0")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    D = json.loads(Path(a.data).read_text())
    rows = []
    for who in ("teacher", "Phase 4.5 car"):
        for d in ("nominal", "gate"):
            sc = D["scores"][f"{who}|{d}"]
            eps = D["episodes"][f"{who}|{d}"]
            far = np.median([min(e["x"] / D["courses"][str(e["course"])]["length"], 1) for e in eps])
            j1 = sum(bool(e["jumps"][0]["landed"]) for e in eps)
            rows.append([who, d, f"<b>{sc['finished_n']}/{sc['runs']}</b>", f"{far:.0%}", endings(sc), f"{j1}/60",
                         f"{sc['landed']}/{sc['jumps']}", f"{sc['lap_s']:.1f} s" if sc["finished_n"] else "–"])
    # where the Phase 4.5 car's runs ended, by turns fully passed
    eps = D["episodes"]["Phase 4.5 car|gate"]
    passed = [sum(1 for _, b, _ in D["courses"][str(e["course"])]["corners"] if b < e["x"]) for e in eps]
    where = [["before or in the first turn", sum(p == 0 for p in passed)], ["in the second turn", sum(p == 1 for p in passed)],
             ["in the third turn", sum(p == 2 for p in passed)], ["later", sum(p > 2 for p in passed)]]
    S = D["sensors"]
    med = lambda k: float(np.median([r[k] for r in S]))  # noqa: E731
    early = sum(r["any"] >= r["brake"] for r in S)
    srows = [["forward lidar (3° down)", f"{med('lidar'):.2f} m", f"{max(r['lidar'] for r in S):.2f} m"],
             ["8×8 time-of-flight", f"{med('tof'):.2f} m", f"{max(r['tof'] for r in S):.2f} m"],
             ["side sensors", f"{med('side'):.2f} m", f"{max(r['side'] for r in S):.2f} m"],
             ["<b>any sensor</b>", f"<b>{med('any'):.2f} m</b>", f"{max(r['any'] for r in S):.2f} m"],
             ["teacher starts slowing", f"<b>{med('brake'):.2f} m</b>", f"{max(r['brake'] for r in S):.2f} m"]]
    kw = {k: v for k, v in cm.CAR.items() if k not in ("loop", "max_steps")}
    kw["dr"] = "gate"
    car = load_car(a.run)
    out = Path("runs/p46-e1")
    g1, g2, g3 = out / "car-10000.gif", out / "car-10000-3d.gif", out / "teacher-10000.gif"
    e1 = rl.gif(car, 10000, g1, "Phase 4.5 car · gate range", **kw)
    t_end = e1["l"] * 0.02
    rl.chase_gif(car, 10000, g2, "Phase 4.5 car · gate range", t_range=(max(t_end - 5.0, 0), t_end), **kw)
    e3 = rl.gif(LoopExpert(), 10000, g3, "teacher · gate range", **kw)
    sections = [
        f"<h2>Result</h2><div class='note'>{a.notes}</div>",
        "<h2>Laps finished (validation loops, 60 laps each)</h2>" + rp.table(
            ["driver", "physics", "laps finished", "median distance (of a lap)", "other endings", "jump 1 landed",
             "all jumps landed", "median lap"], rows)
        + "<p class='caption'>20 validation loops (10000–10019) × 3 starts, one car at a time, Phase 4.5's realistic car "
          "and parts-list sensors. The Phase 4.5 car is your GRU + filter copy trained on straight roller lanes (E5 seed 0, "
          "rebuilt; 84% on straight lanes). The teacher reads the true position. Every lap the teacher finished was "
          "within the gate's time limit by definition (1.5× its own time). Jumps a car never reached count as not landed.</p>",
        "<h2>Where the Phase 4.5 car's runs ended (gate range)</h2>" + rp.table(["ended", "runs (of 60)"], where)
        + "<p class='caption'>Every loop starts with a straight and jump 1, then the first turn.</p>",
        "<h2>How early do the sensors see a turn?</h2>" + rp.table(
            ["", "median distance before the turn begins", "furthest"], srows)
        + f"<p class='caption'>{len(S)} turns that follow at least 11 m of clear straight (no jump, whoops or other turn). "
          "The car stands on the centreline every 0.25 m before the turn; a sensor \"sees\" the turn when any of its "
          "readings differs by more than 0.25 m from the same stretch laid out straight (no noise). 0 m = not before the "
          f"turn begins. The sensors saw the turn before the teacher's slowing point in <b>{early} of {len(S)}</b> turns. "
          "At 7.4 m/s, 1 m is 0.14 s. The turn \"begins\" where the centreline's radius drops under 40 m.</p>",
        "<h2>The drivers on loop 10000</h2><div class='gifs'>"
        + rp.figure(g1, f"Phase 4.5 car, map view: validation loop 10000, start 0, gate-range physics, one run at real "
                        f"speed. Ended: {e1['reason']} after {e1['l'] * 0.02:.1f} s, landed {e1['landed']}/3. Blue trail "
                        "on the ground, orange in the air, dark dots where it slides sideways.")
        + rp.figure(g2, "The same run from behind: its last 5 seconds.")
        + rp.figure(g3, f"Teacher, map view: the same loop and physics, one run at real speed. {e3['reason']} in "
                        f"{e3['l'] * 0.02:.1f} s, landed {e3['landed']}/3.") + "</div>"]
    rp.page("Phase 4.6 E1 turns", "Phase 4.6 · E1 · validation loops 10000–10019", "What breaks on turns?",
            "Phase 4.5's car and the new teacher on loop courses, and how early the parts-list sensors see a turn.",
            sections, OUT, footer="Code: sdc/phase46/e1.py, e1_report.py, teacher.py; sdc/jumps/loop.py")
    print(OUT, e1["reason"], e1["l"], e3["reason"])


if __name__ == "__main__":
    main()
