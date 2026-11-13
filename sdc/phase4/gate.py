"""Phase 4 gate: python -m sdc.phase4.gate --run runs/p4-e5-fb-s2

Test courses 40000-40019 (TEST_SEEDS, never used by experiments), each from the 3 start points
(dr.STARTS) = 180 jumps, nominal physics, one car at a time. Pass: >= 80% landed (entry definition).
The hand-coded expert drives the same 180 jumps for reference (it doesn't affect the verdict).
Writes <run>/gate.json and reports/phase4-gate.html.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from sdc.jumps import course as crs
from sdc.jumps import driver as dr
from sdc.jumps import render, render3d
from sdc.phase4 import report as rp
from sdc.phase4 import train as tr
from sdc.phase4.bc import EXPERT

BAR = 0.80


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    d = Path(a.run)
    pol, m = tr.load(d)
    eps = dr.run(pol, crs.TEST_SEEDS, n_envs=1, starts=dr.STARTS)
    sc = dr.score(eps)
    ex = dr.score(dr.run(dr.ExpertDriver(**EXPERT), crs.TEST_SEEDS, n_envs=1, starts=dr.STARTS))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = dict(run=str(d), commit=commit, courses=crs.TEST_SEEDS, starts=list(dr.STARTS), bar=BAR,
               verdict="PASS" if sc["rate"] >= BAR else "FAIL", score=sc, expert=ex, episodes=eps)
    (d / "gate.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"GATE {res['verdict']}: landed {sc['landed']}/{sc['jumps']} = {sc['rate']:.1%} (bar {BAR:.0%}); "
          f"expert {ex['rate']:.1%}; by kind {sc['by_kind']}; endings {sc['reasons']}")
    g3, g2 = d / "gate3d-40000.gif", d / "gate-40000.gif"
    e3 = render3d.gif(pol, crs.TEST_SEEDS[0], g3, "gate car (copy + PPO + finish bonus)")
    e2 = render.gif(pol, crs.TEST_SEEDS[0], g2, "gate car (copy + PPO + finish bonus)")
    ok = res["verdict"] == "PASS"
    sections = [
        f"""<h2>Verdict</h2><div class="note">
<p><span class="big2" style="color: var({'--s3' if ok else '--threshold'})">{res['verdict']}</span>
&nbsp;{sc['landed']} of {sc['jumps']} test jumps landed: <b>{sc['rate']:.1%}</b> (bar {BAR:.0%}).</p>
<p>Test courses 40000–40019, never used before, each from 3 start points; nominal physics; one car at a time.
The hand-coded expert, which reads the true state, lands {ex['rate']:.1%} of the same jumps.
Validation for this car: {m['final']['rate']:.0%}.</p></div>""",
        "<h2>By jump and ending</h2>" + rp.table(["", "gap", "table", "drop", "finished runs"],
                                                 [["gate car", sc["by_kind"]["gap"], sc["by_kind"]["table"], sc["by_kind"]["drop"], f"{sc['finished']:.0%}"],
                                                  ["expert", ex["by_kind"]["gap"], ex["by_kind"]["table"], ex["by_kind"]["drop"], f"{ex['finished']:.0%}"]])
        + f"<p class='caption'>How the gate car's 60 runs ended: {', '.join(f'{k} {v}' for k, v in sc['reasons'].items())}.</p>",
        "<h2>The car on test course 40000</h2><div class='gifs'>"
        + rp.figure(g3, f"3D chase view, start point 0, one run, real speed. {e3['reason']}, landed {e3['landed']}/3.")
        + rp.figure(g2, f"The same run from the side. {e2['reason']}, landed {e2['landed']}/3.") + "</div>",
    ]
    rp.page("Phase 4 gate", "Phase 4 · gate · test courses 40000–40019", "Landing jumps on new courses",
            "Does the car land at least 80% of jumps on courses it has never seen?", sections, "reports/phase4-gate.html",
            footer=f"python -m sdc.phase4.gate --run {d} · commit {commit}")
    print("reports/phase4-gate.html")


if __name__ == "__main__":
    main()
