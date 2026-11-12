"""Phase 4 E1 report: python -m sdc.phase4.e1_report -> reports/phase4-e1.html (+ runs/p4-e1-summary.json)"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from sdc.jumps import driver as dr
from sdc.jumps import render
from sdc.phase4 import analysis as an
from sdc.phase4 import report as rp
from sdc.phase4 import train as tr

RUNS = [Path(f"runs/p4-e1-s{s}") for s in range(3)]
OUT = Path("reports/phase4-e1.html")


class HoldThrottle:
    """Straight ahead at a fixed throttle: the simplest possible driver."""

    def __init__(self, u=0.5):
        self.u = u

    def __call__(self, b):
        return np.array([[0.0, self.u]] * b.n, np.float32)


def summaries():
    cache = Path("runs/p4-e1-summary.json")
    if cache.exists():
        return json.loads(cache.read_text())
    out = {}
    for name, pol in [("expert", dr.ExpertDriver()), ("expert, throttle held in the air", dr.ExpertDriver(air_control=False)),
                      ("straight, throttle 0.5", HoldThrottle(0.5))]:
        out[name] = an.air_summary(pol)
    for d in RUNS:
        pol, m = tr.load(d)
        out[d.name] = {**an.air_summary(pol), "chosen": m["selection"]["chosen"], "final_rate": m["selection"]["final_rate"]}
    cache.write_text(json.dumps(out, indent=1, default=float))
    return out


def main():
    torch.set_num_threads(1)
    S = summaries()
    runs = [json.loads((d / "metrics.json").read_text()) for d in RUNS]
    best = max(range(3), key=lambda i: S[RUNS[i].name]["score"]["rate"])
    rows = []
    for name in ["expert", "expert, throttle held in the air", "straight, throttle 0.5"] + [d.name for d in RUNS]:
        s = S[name]
        sc = s["score"]
        label = name if not name.startswith("p4-") else f"your PPO, seed {name[-1]}"
        rows.append([label, f"{sc['rate']:.0%}", sc["by_kind"]["gap"], sc["by_kind"]["table"], sc["by_kind"]["drop"],
                     f"{sc['finished']:.0%}", s.get("touch_err_abs", "–"), s.get("corr_err", "–")])
    seeds_rate = [S[d.name]["score"]["rate"] for d in RUNS]
    curve = rp.curve(runs)
    legend = rp.legend([(i, f"seed {i}", f"{S[d.name]['score']['rate']:.0%}") for i, d in enumerate(RUNS)])
    # GIFs: best seed on two validation courses, plus its worst course
    bd = RUNS[best]
    pol, _ = tr.load(bd)
    worst = min(S[bd.name]["eps"] if "eps" in S[bd.name] else [], default=None)
    gifs = []
    for c in (10000, 10004):
        p = bd / f"report-{c}.gif"
        ep = render.gif(pol, c, p, f"your PPO (seed {best})")
        gifs.append(rp.figure(p, f"Seed {best} on validation course {c}: one run, real speed. "
                                 f"{ep['reason']}, landed {ep['landed']}/3."))
    pe = Path("runs/p4-check/report-expert-10000.gif")
    pe.parent.mkdir(parents=True, exist_ok=True)
    render.gif(dr.ExpertDriver(), 10000, pe, "hand-coded expert")
    gifs.append(rp.figure(pe, "The expert on course 10000 for comparison (it reads its true pitch and speed)."))
    del worst
    reasons = {RUNS[i].name: S[RUNS[i].name]["score"]["reasons"] for i in range(3)}
    sections = [
        f"""<h2>Result</h2><div class="note">
<p><span class="big2">{np.mean(seeds_rate):.0%}</span> of validation jumps landed on average over 3 seeds
({", ".join(f"{r:.0%}" for r in seeds_rate)}). Your prediction: 75%, and only partial use of throttle in the air.</p>
<p>{{FINDING}}</p></div>""",
        f"<h2>Learning curves</h2>{curve}{legend}<p class='caption'>Validation landing rate (20 courses, 60 jumps) every 50k steps. "
        "The kept model is each seed's best checkpoint (dot = last evaluation).</p>",
        "<h2>Who lands what</h2>" + rp.table(
            ["driver", "landed", "gap", "table", "drop", "finished", "|pitch − slope| at touchdown", "air throttle ↔ pitch error"], rows)
        + "<p class='caption'>Pitch error at touchdown in degrees (0 = the car matches the ground). Last column: correlation, over every "
          "airborne step, between the throttle change since takeoff and how far the nose is from the landing slope. "
          "Near +1 = steers its pitch with the throttle like the expert; 0 = ignores it.</p>"
        + f"<p class='caption'>Endings per seed: {json.dumps(reasons)}</p>",
        "<h2>The car</h2><div class='gifs'>" + "".join(gifs) + "</div>",
    ]
    rp.page("Phase 4 E1 jumps", "Phase 4 · E1 · validation courses 10000–10019", "Learning to jump from scratch",
            "Can your PPO, with no memory and only real sensors, learn to jump and land?", sections, OUT,
            footer="Code: sdc/jumps/, sdc/phase4/train.py, analysis.py, e1_report.py")
    print(OUT)


if __name__ == "__main__":
    main()
