"""Phase 4.5 E1 report: python -m sdc.phase45.e1_report -> reports/phase45-e1.html

Reads runs/p45-e1/e1.json (python -m sdc.phase45.e1) and draws two 3D GIFs of the gate car on validation
course 10000: as in Phase 4, and with everything on at gate-range physics. Notes come from --notes (HTML).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sdc.jumps import driver as dr
from sdc.jumps import render3d
from sdc.phase4 import report as rp
from sdc.phase4 import train as tr
from sdc.phase45.e1 import CONFIGS

OUT = Path("reports/phase45-e1.html")
WHO = ("gate car", "expert (Phase 4)", "expert (4.5)")


class WithEnv:
    """A policy driven with extra env options (the realistic car)."""

    def __init__(self, pol, **kw):
        self.pol, self.env_kw = pol, {**getattr(pol, "env_kw", {}), **kw}
        if hasattr(pol, "reset"):
            self.reset = pol.reset

    def __call__(self, b):
        return self.pol(b)


def by_phys(eps, key, edges):
    """Landing rate of the gate car's jumps binned by one physics parameter."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        js = [j for e in eps if lo <= e["phys"][key] < hi for j in e["jumps"]]
        rows.append((lo, hi, sum(bool(j["landed"]) for j in js), len(js)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/p45-e1/e1.json")
    ap.add_argument("--run", default="runs/p4-e5-fb-s2")
    ap.add_argument("--notes", default="")
    ap.add_argument("--nogif", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)
    D = json.loads(Path(a.data).read_text())
    rows = []
    for cfg in CONFIGS:
        r = [cfg]
        for w in WHO:
            sc = D[cfg][w]["score"]
            r.append(f"{sc['rate']:.0%}")
        g = D[cfg]["gate car"]["score"]
        r.append(", ".join(f"{k} {v}" for k, v in sorted(g["reasons"].items(), key=lambda kv: -kv[1]) if k != "finished")
                 or "–")
        rows.append(r)
    kinds = [[cfg] + [D[cfg]["gate car"]["score"]["by_kind"][k] for k in ("gap", "table", "drop")]
             for cfg in ("Phase 4 car (as at the gate)", "all physics + sensors", "... + gate-range physics")]
    gate_eps = D["... + gate-range physics"]["gate car"]["episodes"]
    phys_rows = []
    for key, edges, fmt in (("mass", [0.75, 0.93, 1.12, 1.31], "×{:.2f}–{:.2f}"),
                            ("friction", [0.55, 0.78, 1.02, 1.26], "{:.2f}–{:.2f}"),
                            ("delay", [0.0, 0.015, 0.03, 0.046], "{:.3f}–{:.3f} s"),
                            ("noise", [0.5, 1.0, 1.5, 2.01], "×{:.1f}–{:.1f}")):
        for lo, hi, n, t in by_phys(gate_eps, key, edges):
            phys_rows.append([key, fmt.format(lo, hi), f"{n}/{t}", f"{n / max(t, 1):.0%}"])
    sections = [f"<h2>Result</h2><div class='note'>{a.notes}</div>",
                "<h2>What each change costs</h2>" + rp.table(["setup", *WHO, "gate car's other endings (of 60 runs)"], rows)
                + "<p class='caption'>Landed share of 180 jumps (20 validation courses × 3 starts), one car at a time. "
                  "Each row up to 'realistic sensors' turns on one change alone; the last three stack everything. "
                  "'Expert (Phase 4)' is the teacher Phase 4 copied; 'expert (4.5)' is retuned for the new car. "
                  "Both experts read the true state, so the sensor rows don't affect them.</p>",
                "<h2>Gate car by jump kind</h2>" + rp.table(["setup", "gap", "table", "drop"], kinds),
                "<h2>Gate-range physics: what hurts</h2>" + rp.table(["parameter", "range", "landed", "share"], phys_rows)
                + "<p class='caption'>The gate car's jumps in the last row, split by the physics each run drew.</p>"]
    if not a.nogif:
        pol, _ = tr.load(Path(a.run))
        gifs = []
        for tag, kw, label in (("p4", {}, "as in Phase 4"),
                               ("gate", dict(real="all", sensors="noisy", dr="gate"), "all on, gate range")):
            p = Path("runs/p45-e1") / f"gate-car-{tag}-10000.gif"
            ep = render3d.gif(WithEnv(pol, **kw), 10000, p, f"P4 gate car · {label}")
            gifs.append(rp.figure(p, f"3D chase view: the Phase 4 gate car on validation course 10000, {label}; one run, "
                                     f"real speed. {ep['reason']}, landed {ep['landed']}/3."))
        sections.append("<h2>The car</h2><div class='gifs'>" + "".join(gifs) + "</div>")
    rp.page("Phase 4.5 E1 what breaks", "Phase 4.5 · E1 · validation courses 10000–10019", "What breaks on the realistic car?",
            "Phase 4's gate car and the experts, unchanged, on each realistic change and then all of them.", sections, OUT,
            footer="Code: sdc/phase45/e1.py, e1_report.py; sim: sdc/jumps/real.py")
    print(OUT)


if __name__ == "__main__":
    main()
