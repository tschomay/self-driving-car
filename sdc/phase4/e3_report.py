"""Phase 4 E3 report: python -m sdc.phase4.e3_report -> reports/phase4-e3.html (+ runs/p4-e3-summary.json)"""

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
from sdc.phase4 import train_rnn as trr

ARMS = [("mlp", "no memory", tr), ("gru", "your GRU", trr)]
OUT = Path("reports/phase4-e3.html")


def runs(arm):
    return [Path(f"runs/p4-e3-{arm}-s{s}") for s in range(3)]


def summaries():
    cache = Path("runs/p4-e3-summary.json")
    if cache.exists():
        return json.loads(cache.read_text())
    out = {"expert": {**an.air_summary(dr.ExpertDriver()), **an.pitch_probe(dr.ExpertDriver()),
                      **an.steering_stats(dr.ExpertDriver())}}
    for arm, _, mod in ARMS:
        for d in runs(arm):
            pol, m = mod.load(d)
            out[d.name] = {**an.air_summary(pol), **an.pitch_probe(pol), **an.steering_stats(pol),
                           "chosen": m["selection"]["chosen"]}
    cache.write_text(json.dumps(out, indent=1, default=float))
    return out


def main():
    torch.set_num_threads(1)
    S = summaries()
    mean = {arm: np.mean([S[d.name]["score"]["rate"] for d in runs(arm)]) for arm, _, _ in ARMS}
    rows = [["hand-coded expert", f"{S['expert']['score']['rate']:.0%}", S["expert"]["score"]["by_kind"]["gap"],
             S["expert"]["score"]["by_kind"]["table"], S["expert"]["score"]["by_kind"]["drop"],
             S["expert"]["touch_err_abs"], S["expert"]["corr_err"], S["expert"]["steer_change"]]]
    for arm, name, _ in ARMS:
        for d in runs(arm):
            s = S[d.name]
            sc = s["score"]
            rows.append([f"{name}, seed {d.name[-1]}", f"{sc['rate']:.0%}", sc["by_kind"]["gap"], sc["by_kind"]["table"],
                         sc["by_kind"]["drop"], s.get("touch_err_abs"), s.get("corr_err"), s["steer_change"]])
    curves = []
    for k, (arm, name, _) in enumerate(ARMS):
        for d in runs(arm):
            f = d / "rescore.json"
            ev = json.loads(f.read_text()) if f.exists() else json.loads((d / "metrics.json").read_text())["evals"]
            curves.append(dict(evals=ev, _color=k))
    probe_rows = [[f"{name}, seed {d.name[-1]}", S[d.name]["r2_sensors"], S[d.name].get("r2_memory", "–"),
                   S[d.name].get("r2_both", "–")] for arm, name, _ in ARMS for d in runs(arm)]
    best = {arm: max(runs(arm), key=lambda d: S[d.name]["score"]["rate"]) for arm, _, _ in ARMS}
    figs = []
    for arm, name, mod in ARMS:
        pol, _ = mod.load(best[arm])
        p = best[arm] / "report-10004.gif"
        ep = render.gif(pol, 10004, p, f"{name} (seed {best[arm].name[-1]})")
        figs.append(rp.figure(p, f"{name.capitalize()}, its best seed, on validation course 10004: one run, real speed. "
                                 f"{ep['reason']}, landed {ep['landed']}/3."))
    sections = [
        f"""<h2>Result</h2><div class="note">
<p><span class="big2">{mean['gru']:.0%}</span> landed with your GRU vs <b>{mean['mlp']:.0%}</b> without memory
(3 seeds each, 180 validation jumps per seed). Your prediction: 70%, and it would use the throttle in the air.</p>
<p>No clear gain from memory. One car at a time, the GRU seeds land 65/61/43% vs 57/51/42% without memory; scored in batches of ten
(float rounding alone changes outcomes), it is 57/58/49% vs 69/61/41%. The seed-to-seed spread (about 20 points) is bigger than
any difference between the two cars.</p>
<p><b>What changed:</b> the steering is calm now. The change per step on the flats is 0.00–0.07, against 1.37 in E1.
<b>What didn't:</b> neither car uses the throttle in the air. Its throttle barely moves after takeoff (0.02–0.19 vs the
expert's 0.37), and the link to the pitch error is about zero (−0.36 to +0.14, expert +0.81). The memory does hold the
pitch: a straight-line read-out of the GRU's memory recovers it with R² up to 0.84. But the sensors already give most of it
(0.8–0.9, from the downward rays seeing the ground), and the policy doesn't act on either. The runs now end
"off the lane" or "stuck" (casing a gap and stopping) more than rolled over. The learning curves swing wildly from one
checkpoint to the next (0% to 60%), in both cars.</p>
<p><b>Reading:</b> knowing the pitch wasn't the missing piece. PPO from scratch never finds out that throttle in the air
pays off: it's a rare, precise skill, and random exploration almost never stumbles on it. That is Phase 3.5's lesson
("it never tried going faster"), and the planned E4 (start from the expert, then fine-tune with low noise) is aimed at it.</p></div>""",
        f"<h2>Learning curves</h2>{rp.curve(curves)}"
        + rp.legend([(0, "no memory (3 seeds)", f"{mean['mlp']:.0%}"), (1, "your GRU (3 seeds)", f"{mean['gru']:.0%}")])
        + "<p class='caption'>Validation landing rate of every 50k-step checkpoint (3-start re-score, 180 jumps). "
          "Each seed keeps its best checkpoint.</p>",
        "<h2>Who lands what</h2>" + rp.table(
            ["driver", "landed", "gap", "table", "drop", "|pitch − slope|°", "air throttle ↔ pitch error", "steering change"],
            rows) + "<p class='caption'>Steering change: mean |steer now − last step| on the flats (E1 seed 0: 1.37).</p>",
        "<h2>Where is the pitch written down?</h2>" + rp.table(
            ["driver", "from current sensors", "from the memory", "both"], probe_rows)
        + f"<p class='caption'>R² of a straight-line read-out of the car's true pitch in the air (1 = exactly, 0 = not at all), "
          f"fitted on half the courses and tested on the other half. For the expert's flights the sensors alone give "
          f"{S['expert']['r2_sensors']}: the downward rays see the ground and give the pitch away.</p>",
        "<h2>The cars</h2><div class='gifs'>" + "".join(figs) + "</div>",
    ]
    rp.page("Phase 4 E3 memory", "Phase 4 · E3 · validation courses 10000–10019", "Does memory help in the air?",
            "Your GRU vs no memory, everything else equal.", sections, OUT,
            footer="Code: sdc/phase4/my_gru.py (yours), train_rnn.py, analysis.py, e3_report.py")
    print(OUT)


if __name__ == "__main__":
    main()
