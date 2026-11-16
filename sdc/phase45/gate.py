"""Phase 4.5 gate: python -m sdc.phase45.gate --run runs/p45-e6-s1

Protocol (owner, 2026-11-16, fixed before any test result): the roller versions of test courses 45000-45019 (never used
before), 3 starts each (180 jumps), gate-range physics from physics seed 45, parts-list sensors, one car at a time.
Each test course is first crawled at the same physics by the sensor-only crawl driver; its map comes from the estimator
fitted on practice courses only (runs/p45-maps-estimator.json). Pass: >= 80% landed (Phase 4's definition).
The map-following expert drives the same 180 jumps for reference. Writes <run>/gate.json, runs/p45-maps-test.json and
reports/phase45-gate.html.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from sdc.jumps import course as crs
from sdc.jumps import driver as dr
from sdc.jumps import render3d
from sdc.phase4 import report as rp
from sdc.phase4 import train_rnn as trr
from sdc.phase45 import maps as mp
from sdc.phase45 import outlap as ol
from sdc.phase45.common import PHYS_SEED_TEST, TEST_SEEDS
from sdc.phase45.e1_report import WithEnv

BAR = 0.80
MAP = Path("runs/p45-maps-test.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/p45-e6-s1")
    a = ap.parse_args()
    torch.set_num_threads(1)
    if not MAP.exists():   # 1) out-laps of the test courses, at the gate's physics
        rows = ol.walk_collect(TEST_SEEDS, starts=dr.STARTS, dr="gate", procs=4, phys_seed=PHYS_SEED_TEST)
        model = json.loads(Path("runs/p45-maps-estimator.json").read_text())
        MAP.write_text(json.dumps(mp.build(rows, model)))
    env = dict(real="all", sensors="parts", attitude=True, kinds=crs.KINDS45, dr="gate", phys_seed=PHYS_SEED_TEST,
               track_map=str(MAP))
    pol, m = trr.load(Path(a.run))
    eps = dr.run(WithEnv(pol, **env), TEST_SEEDS, n_envs=1, starts=dr.STARTS)   # 2) the gate car
    sc = dr.score(eps)
    ex = dr.score(dr.run(dr.ExpertDriver(plan=False, v_target=7.4, kd=1.0, use_map=True, plan_offset=0.6), TEST_SEEDS,
                         n_envs=1, starts=dr.STARTS, **env))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = dict(run=a.run, commit=commit, bar=BAR, verdict="PASS" if sc["rate"] >= BAR else "FAIL", score=sc,
               expert=ex, validation=m["final"]["rate"], episodes=eps)
    Path(a.run, "gate.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"GATE {out['verdict']}: {sc['landed']}/{sc['jumps']} = {sc['rate']:.1%} (bar {BAR:.0%}); "
          f"expert {ex['rate']:.1%}; by kind {sc['by_kind']}; endings {sc['reasons']}")
    g = Path(a.run, "gate-45000.gif")
    ep = render3d.gif(WithEnv(pol, **env), 45000, g, "Phase 4.5 gate car · test course 45000")
    rows = [["gate car", f"{sc['rate']:.1%}", *[sc["by_kind"].get(k, "–") for k in ("double", "table", "drop")],
             f"{sc['finished']:.0%}"],
            ["expert + map (reads the true state)", f"{ex['rate']:.1%}", *[ex["by_kind"].get(k, "–") for k in
                                                                         ("double", "table", "drop")], f"{ex['finished']:.0%}"]]
    sections = [f"""<h2>Verdict</h2><div class="note"><p><span class="big2">{out['verdict']}</span>: the gate car landed
<b>{sc['landed']}/{sc['jumps']} = {sc['rate']:.1%}</b> of test jumps (bar {BAR:.0%}). Validation: {m['final']['rate']:.0%}.
Endings: {sc['reasons']}.</p></div>""",
                "<h2>Test courses 45000–45019 (rollers), gate-range physics</h2>" + rp.table(
                    ["driver", "landed", "doubles", "tables", "drops", "courses finished"], rows)
                + "<p class='caption'>180 jumps (20 courses × 3 starts), physics seed 45, one car at a time. Each course "
                  "crawled first at the same physics to build its map. Commit " + commit + ".</p>",
                "<h2>The car</h2><div class='gifs'>" + rp.figure(
                    g, f"3D chase view: the gate car on test course 45000, start 0, one run, real speed. {ep['reason']}, "
                       f"landed {ep['landed']}/3.") + "</div>"]
    rp.page("Phase 4.5 gate", "Phase 4.5 · gate · test courses 45000–45019", "Phase 4.5 gate check",
            "Does the realistic car land at least 80% of jumps on fresh test courses, with physics outside its training "
            "range?", sections, "reports/phase45-gate.html", footer="Code: sdc/phase45/gate.py")


if __name__ == "__main__":
    main()
