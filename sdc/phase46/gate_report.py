"""Phase 4.6 gate report: python -m sdc.phase46.gate_report --notes "<html>" -> reports/phase46-gate.html

Reads runs/p46-gate/gate.json (python -m sdc.phase46.gate) and draws map-view GIFs of the gate car on two test runs.
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
from sdc.phase46.e3_report import endings
from sdc.phase46.ensemble import Ensemble

OUT = Path("reports/phase46-gate.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="")
    ap.add_argument("--gifs", nargs="*", default=["46000:0.0", "46008:0.0"])
    ap.add_argument("--gate", default="runs/p46-gate")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--title", default="Gate check")
    a = ap.parse_args()
    torch.set_num_threads(1)
    gdir = Path(a.gate)
    g = json.loads((gdir / "gate.json").read_text())
    c, t = g["car_score"], g["teacher"]
    rows = [[f"gate car (average of {len(g['runs'])} students)", f"<b>{g['with_out_lap']}/60</b>", f"{g['timed_lap_only']}/60",
             endings(c), f"{c['rate']:.0%}", f"{c['lap_s']:.1f} s"],
            ["standard teacher (true track, reference)", "–", f"{t['finished_n']}/60", endings(t), f"{t['rate']:.0%}",
             f"{t['lap_s']:.1f} s"]]
    env = {k: v for k, v in g["env"].items() if k not in ("loop", "max_steps")}
    figs = []
    for spec in a.gifs:
        course, x0 = int(spec.split(":")[0]), float(spec.split(":")[1])
        p = gdir / f"gate-{course}-{x0:+.1f}.gif"
        ep = rl.gif(Ensemble(g["runs"]), course, p, "Gate car · test loop · gate range", x0=x0, **env)
        figs.append(rp.figure(p, f"Gate car: test loop {course}, start {x0:+.1f} m, gate-range physics (seed 46), "
                                 f"one run at real speed. Ended: {ep['reason']} after {ep['l'] * 0.02:.1f} s, "
                                 f"landed {ep['landed']}/3."))
    sections = [f"<h2>Verdict</h2><div class='note'>{a.notes}</div>",
                "<h2>Test loops 46000–46019 × 3 starts (60 runs), gate-range physics</h2>" + rp.table(
                    ["driver", "finished (gate: out-lap + timed lap)", "timed lap only", "other endings", "jumps landed",
                     "median lap"], rows)
                + f"<p class='caption'>Gate: at least 48 of 60. Out-laps failed: {g['out_laps_failed']} "
                  f"({', '.join(f'{k} {v}' for k, v in g['out_lap_reasons'].items()) or 'none'}). A lap counts if "
                  "finished within 1.5× the teacher's time on the same run (the teacher's median when it failed).</p>",
                "<h2>The gate car on two test runs</h2><div class='gifs'>" + "".join(figs) + "</div>"]
    rp.page("Phase 4.6 gate", "Phase 4.6 · gate · test loops 46000–46019", a.title,
            "The averaged students on loops never seen before.", sections, Path(a.out),
            footer="Code: sdc/phase46/gate.py, gate_report.py, ensemble.py")
    print(a.out)


if __name__ == "__main__":
    main()
