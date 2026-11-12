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


def baselines():
    cache = Path("runs/p4-baselines.json")
    if cache.exists():
        return json.loads(cache.read_text())
    out = {}
    for name, pol in [("expert", dr.ExpertDriver()), ("expert, throttle held in the air", dr.ExpertDriver(air_control=False)),
                      ("straight, throttle 0.5", HoldThrottle(0.5))]:
        out[name] = an.air_summary(pol)
    cache.write_text(json.dumps(out, indent=1, default=float))
    return out


def summaries():
    cache = Path("runs/p4-e1-summary.json")
    if cache.exists():
        return json.loads(cache.read_text())
    out = dict(baselines())
    for d in RUNS:
        pol, m = tr.load(d)
        out[d.name] = {**an.air_summary(pol), "chosen": m["selection"]["chosen"], "final_rate": m["selection"]["final_rate"]}
    cache.write_text(json.dumps(out, indent=1, default=float))
    return out


def main():
    torch.set_num_threads(1)
    S = summaries()
    runs = []
    for i, d in enumerate(RUNS):   # curves from the 3-start re-score (180 jumps per point), not the noisy 60-jump one
        runs.append(dict(evals=json.loads((d / "rescore.json").read_text()), _color=i))
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
    ends = ["finished", "rolled over", "off the lane", "stuck"]
    erows = []
    for name in ["expert", "expert, throttle held in the air", "straight, throttle 0.5"] + [d.name for d in RUNS]:
        r = S[name]["score"]["reasons"]
        label = name if not name.startswith("p4-") else f"your PPO, seed {name[-1]}"
        erows.append([label] + [r.get(k, 0) for k in ends])
    sr = np.mean(seeds_rate)
    finding = (f"It learned to drive at the jumps and get airborne, but not much more: the simplest driver there is, straight ahead "
               f"at half throttle, lands {S['straight, throttle 0.5']['score']['rate']:.0%}. Two things hold it back. "
               "It barely uses the throttle in the air: the link between its throttle and how far its nose is off the landing slope "
               f"is {np.mean([S[d.name]['corr_err'] for d in RUNS]):.2f} (expert {S['expert']['corr_err']:.2f}), so it lands "
               f"{np.mean([S[d.name]['touch_err_mean'] for d in RUNS]):+.0f}° off the slope on average, nose high (expert "
               f"{S['expert']['touch_err_mean']:+.0f}°). And it often can't drive straight after a landing: about one run in "
               "four ends off the lane, and as many in a rollover. The curves peak around 550k–750k steps and then slide, "
               "as in Phase 3.")
    del sr
    sections = [
        f"""<h2>Result</h2><div class="note">
<p><span class="big2">{np.mean(seeds_rate):.0%}</span> of validation jumps landed on average over 3 seeds
({", ".join(f"{r:.0%}" for r in seeds_rate)}); the expert lands {S['expert']['score']['rate']:.0%}. Your prediction: 75%, and only partial use of throttle in the air.</p>
<p>{finding}</p>
<p class="caption">Scores are over 180 jumps: each of the 20 validation courses driven from 3 start points 0.3 m apart.
Landings are chaotic: the same seed-0 network scored 55% when its 20 runs were done as a batch and 40% one car at a
time, from float rounding alone. One pass over 60 jumps is noisy, so experiments from here on use 3 starts.</p></div>""",
        f"<h2>Learning curves</h2>{curve}{legend}<p class='caption'>Validation landing rate (20 courses, 60 jumps) every 50k steps. "
        "The kept model is each seed's best checkpoint (dot = last evaluation).</p>",
        "<h2>Who lands what</h2>" + rp.table(
            ["driver", "landed", "gap", "table", "drop", "finished", "|pitch − slope| at touchdown", "air throttle ↔ pitch error"], rows)
        + "<p class='caption'>Pitch error at touchdown in degrees (0 = the car matches the ground). Last column: correlation, over every "
          "airborne step, between the throttle change since takeoff and how far the nose is from the landing slope. "
          "Near +1 = steers its pitch with the throttle like the expert; 0 = ignores it.</p>"
        + "<h2>How runs end</h2>" + rp.table(["driver"] + ends, erows)
        + "<p class='caption'>60 runs per driver (20 courses × 3 starts).</p>",
        "<h2>The car</h2><div class='gifs'>" + "".join(gifs) + "</div>",
    ]
    rp.page("Phase 4 E1 jumps", "Phase 4 · E1 · validation courses 10000–10019", "Learning to jump from scratch",
            "Can your PPO, with no memory and only real sensors, learn to jump and land?", sections, OUT,
            footer="Code: sdc/jumps/, sdc/phase4/train.py, analysis.py, e1_report.py")
    print(OUT)


if __name__ == "__main__":
    main()
