"""Phase 3.5 gate report: python -m sdc.phase35.gate_report -> reports/phase35-gate.html"""

import base64
import json
from collections import defaultdict
from pathlib import Path

import torch

from sdc.phase3 import robust
from sdc.phase35 import gate as g
from sdc.phase35 import selfmap as sm
from sdc.phase35.e5_report import BINS, grip_bin
from sdc.racer import dynamics as dy
from sdc.racer import render
from sdc.racer import track as trk


def main():
    torch.set_num_threads(1)
    r = json.loads(Path("runs/p35-gate/gate.json").read_text())
    val = json.loads(Path("runs/p35-e5-lo-s0/eval_probe_dense.json").read_text())
    by = defaultdict(list)
    for t in r["gate"]["tracks"]:
        by[grip_bin(t["physics"]["grip"])].append(t["finished"])
    grip = "".join(f"<td>{sum(by[b])}/{len(by[b])}</td>" for b in BINS)
    fails = [t for t in r["gate"]["tracks"] if not t["finished"]]
    n_out = sum(t["laps"] < 0.5 for t in fails)
    seed = trk.TEST35_SEEDS[0]
    rec = robust.laps(g.candidate(r["run"]), seeds=[seed], record=True, outlap=True, **sm.CAR)
    gp = Path(f"runs/p35-gate/lap-{seed}.gif")
    render.lap_gif(seed, rec["trajectories"][0], rec["tracks"][0], "gate car: probes + own map (test track)", gp,
                   dirt=dy.dirt_mask(seed))
    gif = base64.b64encode(gp.read_bytes()).decode()
    css = Path("reports/phase35-e2.html").read_text().split("<style>")[1].split("</style>")[0]
    lap, hand = r["nominal"]["mean_lap_time"], r["hand_nominal"]["mean_lap_time"]
    page = f"""<title>Phase 3.5 gate</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}
img.gif {{ width:100%; max-width:480px; border-radius:6px; border:1px solid var(--line) }}
.verdict {{ font:600 2rem var(--mono); color:var(--hand) }}
</style>
<main>
<header><div class="eyebrow">Phase 3.5 · gate · test tracks 30000–30019 · physics seed 67890</div>
<h1>Phase 3.5 gate: {'pass' if r['passed'] else 'fail'}</h1>
<p class="cap">One look at 20 tracks no experiment has used. The car drives with no sim-only information: it probes the surface on an out-lap, builds its own grip map, and drives the timed lap on it. Both laps must be finished.</p></header>

<section class="note"><div class="verdict">{'PASS' if r['passed'] else 'FAIL'}</div>
<div class="wide"><table><tr><th>check</th><th>result</th><th>bar</th><th></th></tr>
<tr><td>finished, gate-range physics (100 attempts)</td><td>{r['gate']['completion']:.0%}</td><td>≥ 85%</td><td>{'pass' if r['reliability_pass'] else 'fail'}</td></tr>
<tr><td>timed lap at nominal vs. hand-coded (paired)</td><td>{lap:.1f} s vs. {hand:.1f} s = {r['paired_ratio']:.2f}×</td><td>≤ 1.20×</td><td>{'pass' if r['speed_pass'] else 'fail'}</td></tr>
</table></div>
<p class="cap">Nominal: {r['nominal']['completion']:.0%} of 20 test tracks finished. The hand-coded driver (knows grip, mass and dirt) on the same 100 gate-range attempts: {r['hand_gate_completion']:.0%}. Validation, for comparison: {val['gate']['completion']:.0%} and {val['nominal']['mean_lap_time']:.1f} s.</p></section>

<section><h2>By grip</h2>
<p class="cap">Finished / attempts in the gate-range draws, by the episode's grip. Training covered 0.7–1.1×. Of {len(fails)} failures, {n_out} were on the out-lap and {len(fails) - n_out} on the timed lap.</p>
<div class="wide"><table><tr>{"".join(f"<th>{b}×</th>" for b in BINS)}</tr><tr>{grip}</tr></table></div></section>

<section><h2>Where Phase 3 left off</h2>
<p>Phase 3's gate car finished 96% at a standing-start lap of 41.9 s, 1.65× the hand-coded driver. This one finishes {r['gate']['completion']:.0%} with a timed lap of {r['paired_ratio']:.2f}× the hand-coded driver. Three things got it there: starting from the expert's fast driving (E4), lower exploration noise (E4), and a grip map built by probing the surface on the out-lap (E7–E9).</p></section>

<section><h2>Out-lap and timed lap on a test track</h2>
<p class="cap">Test track {seed}, nominal physics, real speed. Short red pedal pulses on the out-lap are the probes; the second lap is the timed lap on the map it built. Brown: dirt.</p>
<img class="gif" alt="Gate car lap GIF" src="data:image/gif;base64,{gif}"></section>
<p class="cap">Command: python -m sdc.phase35.gate --run runs/p35-e5-lo-s0 (commit c4c5b91). Code review before the run: no oracle leakage. Report: sdc/phase35/gate_report.py.</p>
</main>
"""
    Path("reports/phase35-gate.html").write_text(page)
    print("reports/phase35-gate.html", {b: f"{sum(by[b])}/{len(by[b])}" for b in BINS}, len(fails), n_out)


if __name__ == "__main__":
    main()
