"""Phase 4 E4 report: python -m sdc.phase4.e4_report -> reports/phase4-e4.html (+ runs/p4-e4-summary.json)"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from sdc.jumps import driver as dr
from sdc.jumps import render, render3d
from sdc.phase4 import analysis as an
from sdc.phase4 import report as rp
from sdc.phase4 import train as tr
from sdc.phase4.bc import EXPERT

OUT = Path("reports/phase4-e4.html")
ARMS = [("bc", "copy of the expert"), ("ft", "copy + your PPO")]


def runs(arm):
    return [Path(f"runs/p4-e4-{arm}-s{s}") for s in range(3)]


def summaries():
    cache = Path("runs/p4-e4-summary.json")
    if cache.exists():
        return json.loads(cache.read_text())
    ex = dr.ExpertDriver(**EXPERT)
    out = {"expert": {**an.air_summary(ex), **an.steering_stats(ex)}}
    for arm, _ in ARMS:
        for d in runs(arm):
            pol, m = tr.load(d)
            out[d.name] = {**an.air_summary(pol), **an.steering_stats(pol)}
    cache.write_text(json.dumps(out, indent=1, default=float))
    return out


def main():
    torch.set_num_threads(1)
    S = summaries()
    e3 = json.loads(Path("runs/p4-e3-summary.json").read_text())
    mean = {arm: np.mean([S[d.name]["score"]["rate"] for d in runs(arm)]) for arm, _ in ARMS}
    rows = [["expert (steady 7.1 m/s)", f"{S['expert']['score']['rate']:.0%}", S["expert"]["score"]["by_kind"]["gap"],
             S["expert"]["score"]["by_kind"]["table"], S["expert"]["score"]["by_kind"]["drop"],
             S["expert"]["touch_err_abs"], S["expert"]["corr_err"], S["expert"]["steer_change"]]]
    for arm, name in ARMS:
        for d in runs(arm):
            s = S[d.name]
            sc = s["score"]
            rows.append([f"{name}, seed {d.name[-1]}", f"{sc['rate']:.0%}", sc["by_kind"]["gap"], sc["by_kind"]["table"],
                         sc["by_kind"]["drop"], s.get("touch_err_abs"), s.get("corr_err"), s["steer_change"]])
    e3m = np.mean([e3[f"p4-e3-mlp-s{i}"]["score"]["rate"] for i in range(3)])
    rows.append(["E3, no memory (mean of 3)", f"{e3m:.0%}", "", "", "", "", "", ""])
    ends = ["finished", "rolled over", "off the lane", "stuck"]
    erows = [[n] + [S[k]["score"]["reasons"].get(e, 0) for e in ends] for n, k in
             [("expert", "expert")] + [(f"{name}, seed {d.name[-1]}", d.name) for arm, name in ARMS for d in runs(arm)]]
    curves = []
    for i, d in enumerate(runs("ft")):
        f = d / "rescore.json"
        ev = json.loads(f.read_text()) if f.exists() else json.loads((d / "metrics.json").read_text())["evals"]
        curves.append(dict(evals=ev, _color=i))
    best = max(runs("ft"), key=lambda d: S[d.name]["score"]["rate"])
    pol, _ = tr.load(best)
    g3 = best / "report3d-10004.gif"
    ep3 = render3d.gif(pol, 10004, g3, f"copy + your PPO (seed {best.name[-1]})")
    g2 = best / "report-10004.gif"
    ep2 = render.gif(pol, 10004, g2, f"copy + your PPO (seed {best.name[-1]})")
    figs = [rp.figure(g3, f"3D chase view: its best seed on validation course 10004, one run, real speed. "
                          f"{ep3['reason']}, landed {ep3['landed']}/3."),
            rp.figure(g2, f"The same run from the side. {ep2['reason']}, landed {ep2['landed']}/3.")]
    sections = [
        f"""<h2>Result</h2><div class="note">
<p>The copy lands <span class="big2">{mean['bc']:.0%}</span>; after your PPO's fine-tune
<b>{mean['ft']:.0%}</b> (3 seeds each, 180 validation jumps per seed). Your prediction: 80%, and fine-tuning raises it slightly.
E3's best from scratch: {e3m:.0%}. Gate: 80% (on the test courses).</p>
<p><b>The copy works, and it uses the throttle in the air.</b> Its throttle moves with the pitch error (link 0.66–0.69;
the expert 0.78, E3 about 0), and it lands 6–9° nose down where the expert lands within 1°. It copies the expert's
pitch control from its sensors alone, the skill PPO never found from scratch. The failures that remain are mostly
"off the lane" after landings, and the gap jumps.</p>
<p><b>Fine-tuning didn't help.</b> It gained 2–6 points in the first 150k steps, then slid to 47–63% by 1M steps
(curves above). Each seed's kept checkpoint is the copy itself (seed 0) or the 150k one (seeds 1–2), so the
fine-tuned numbers in the table are the copy plus noise. What PPO changed (seed 1, the copy → 1M steps): approach
speed 6.9 → 7.5 m/s, throttle on the ground 0.74 → 0.88, runs off the lane on 10 courses 2 → 8. It trades landings for
speed: metres reached sooner are worth a little more with discounting, and it found that before it found how to keep
landing at that speed. The same slide as Phase 3, where reliability peaked mid-training and was then traded for speed.</p>
<p>Validation is about 4 points above the 80% gate bar on one car at a time, closer than it looks: Phase 3's validation ran
about 4 points above its test.</p></div>""",
        "<h2>Fine-tune learning curves</h2>" + rp.curve(curves)
        + rp.legend([(i, f"seed {i}", f"{S[d.name]['score']['rate']:.0%}") for i, d in enumerate(runs("ft"))])
        + "<p class='caption'>Validation landing rate of every 50k-step checkpoint (3-start re-score, 180 jumps); step 0 is the copy. "
          "Each seed keeps its best checkpoint.</p>",
        "<h2>Who lands what</h2>" + rp.table(
            ["driver", "landed", "gap", "table", "drop", "|pitch − slope|°", "air throttle ↔ pitch error", "steering change"], rows)
        + "<p class='caption'>One car at a time, 180 validation jumps each.</p>",
        "<h2>How runs end</h2>" + rp.table(["driver"] + ends, erows),
        "<h2>The car</h2><div class='gifs'>" + "".join(figs) + "</div>",
    ]
    rp.page("Phase 4 E4 clone", "Phase 4 · E4 · validation courses 10000–10019", "Copy the expert, then fine-tune",
            "Does starting from a copy of the expert get your PPO to the jumps?", sections, OUT,
            footer="Code: sdc/phase4/bc.py, train.py, e4_report.py, sdc/jumps/render3d.py")
    print(OUT)


if __name__ == "__main__":
    main()
