"""Phase 4 E5 report: python -m sdc.phase4.e5_report -> reports/phase4-e5.html (+ runs/p4-e5-summary.json)"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from sdc.jumps import course as crs
from sdc.jumps import driver as dr
from sdc.jumps import render3d
from sdc.phase4 import analysis as an
from sdc.phase4 import report as rp
from sdc.phase4 import train as tr
from sdc.phase4.bc import EXPERT

OUT = Path("reports/phase4-e5.html")
GROUPS = [("bc", "copy of the expert", "p4-e4-bc"), ("ft", "copy + PPO (E4)", "p4-e4-ft"),
          ("fb", "copy + PPO + finish bonus (E5)", "p4-e5-fb")]


def runs(prefix):
    return [Path(f"runs/{prefix}-s{s}") for s in range(3)]


def final_policy(d):
    """The run's last checkpoint (end of training), not the kept best one."""
    pol, m = tr.load(d)
    last = max((d / "ckpt").glob("*.pt"), key=lambda q: int(q.stem))
    pol.net.load_state_dict(torch.load(last))
    return pol


def summaries():
    cache = Path("runs/p4-e5-summary.json")
    if cache.exists():
        return json.loads(cache.read_text())
    out = {"expert": {**an.speed_stats(dr.ExpertDriver(**EXPERT)), **dr.score(dr.run(dr.ExpertDriver(**EXPERT),
                                                                                    crs.VAL_SEEDS, starts=dr.STARTS))}}
    for key, _, prefix in GROUPS:
        for d in runs(prefix):
            pol, m = tr.load(d)
            out[d.name] = {**an.speed_stats(pol), **an.steering_stats(pol), "rate": m["final"]["rate"],
                           "reasons": m["final"]["reasons"]}
            if key != "bc":
                fp = final_policy(d)
                sc = dr.score(dr.run(fp, crs.VAL_SEEDS, starts=dr.STARTS))
                out[d.name + "-final"] = {**an.speed_stats(fp), "rate": sc["rate"], "reasons": sc["reasons"]}
    cache.write_text(json.dumps(out, indent=1, default=float))
    return out


def main():
    torch.set_num_threads(1)
    S = summaries()
    ex = S["expert"]
    rows = [["expert (steady 7.1 m/s)", f"{ex['rate']:.0%}", "", ex["approach_speed"], ex["finish_time"]]]
    for key, name, prefix in GROUPS:
        for d in runs(prefix):
            s = S[d.name]
            f = S.get(d.name + "-final")
            rows.append([f"{name}, seed {d.name[-1]}", f"{s['rate']:.0%}", f"{f['rate']:.0%}" if f else "–",
                         f"{s['approach_speed']} → {f['approach_speed']}" if f else s["approach_speed"],
                         f"{s['finish_time']} → {f['finish_time']}" if f else s["finish_time"]])
    curves = []
    for k, (key, name, prefix) in enumerate(GROUPS[1:]):
        for d in runs(prefix):
            curves.append(dict(evals=json.loads((d / "rescore.json").read_text()), _color=k))
    mean = {key: np.mean([S[d.name]["rate"] for d in runs(prefix)]) for key, _, prefix in GROUPS}
    mfin = {key: np.mean([S[d.name + "-final"]["rate"] for d in runs(prefix)]) for key, _, prefix in GROUPS[1:]}
    best = max(runs("p4-e5-fb"), key=lambda d: S[d.name]["rate"])
    pol, _ = tr.load(best)
    g3 = best / "report3d-10000.gif"
    ep = render3d.gif(pol, 10000, g3, f"copy + PPO + finish bonus (seed {best.name[-1]})")
    sections = [
        f"""<h2>Result</h2><div class="note">
<p>With the finish bonus, the kept checkpoints land <span class="big2">{mean['fb']:.0%}</span> and the end of training
<b>{mfin['fb']:.0%}</b> (E4 without the bonus: kept {mean['ft']:.0%}, end {mfin['ft']:.0%}; the copy {mean['bc']:.0%}).
Your prediction: slides a little but speed improves, 85–88%.</p>
<p><b>The bonus cuts the slide.</b> By the end of training the fine-tunes land 89/72/75% (E4: 59/58/47%), and the
best kept checkpoints 92/94/96%, against the copy's 87–92%. Seed 2 kept improving until 400k steps. It still slides
after its peak, but by about half as much.</p>
<p><b>Speed didn't change.</b> The time to finish a course is about 11.6–12.2 s for every car, the expert included (11.7 s),
and approach speeds stay at 6.7–7.2 m/s. The jumps are built for 6–7.5 m/s, so there's little time to gain, and E4's extra
throttle (0.85–0.89 at the end) bought nothing: it ran off the lane instead.</p>
<p><b>What still goes wrong:</b> losing the lane after landings (off the lane), and, for one seed, stopping after casing
a gap (stuck). Steering is still smooth (0.03–0.08 change per step).</p></div>""",
        "<h2>Fine-tune learning curves</h2>" + rp.curve(curves)
        + rp.legend([(0, "E4: no finish bonus", f"end {mfin['ft']:.0%}"), (1, "E5: finish bonus 100", f"end {mfin['fb']:.0%}")])
        + "<p class='caption'>Validation landing rate of every 50k-step checkpoint (180 jumps); step 0 is the copy.</p>",
        "<h2>Landing vs speed</h2>" + rp.table(
            ["driver", "landed (kept)", "landed (end)", "approach m/s (kept → end)", "finish time s (kept → end)"], rows)
        + "<p class='caption'>Approach speed: the last metre before each kicker. Finish time: mean over finished runs (20 courses, one start).</p>",
        "<h2>The car</h2><div class='gifs'>" + rp.figure(g3, f"3D chase view: its best seed on validation course 10000, one run, real speed. "
                                                            f"{ep['reason']}, landed {ep['landed']}/3.") + "</div>",
    ]
    rp.page("Phase 4 E5 finish bonus", "Phase 4 · E5 · validation courses 10000–10019", "A finish bonus to lock in landings",
            "Does a bonus for finishing the course stop fine-tuning from trading landings for speed?", sections, OUT,
            footer="Code: sdc/phase4/e5_runs.sh, e5_report.py, analysis.py")
    print(OUT)


if __name__ == "__main__":
    main()
