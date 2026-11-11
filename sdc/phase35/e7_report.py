"""Phase 3.5 E7 report: python -m sdc.phase35.e7_report -> reports/phase35-e7.html"""

import base64
import json
from pathlib import Path

import torch

from sdc.phase3 import robust
from sdc.phase35 import probe
from sdc.phase35 import selfmap as sm
from sdc.phase35.e5_report import BINS
from sdc.phase35.e6_report import by_grip
from sdc.racer import dynamics as dy
from sdc.racer import render


def main():
    torch.set_num_threads(1)
    d = Path("runs/p35-e5-lo-s0")
    e5 = json.loads((d / "eval.json").read_text())
    e6 = json.loads((d / "eval_braketest.json").read_text())
    e7 = json.loads((d / "eval_probe.json").read_text())
    cal = json.loads(probe.CAL.read_text())
    rows = [("true grip and map", e5["oracle"]), ("E5: window estimate", e5["self"]), ("E6: one brake test", e6),
            ("E7: probes every 6 m", e7)]
    trs = "".join(f"<tr><td>{n}</td><td>{x['gate']['completion']:.0%}</td><td>{x['nominal']['completion']:.0%}</td>"
                  f"<td>{x['nominal']['mean_lap_time']:.1f} s</td></tr>" for n, x in rows)
    grs = "".join(f"<tr><td>{n}</td>{by_grip(x['gate']['tracks'])}</tr>" for n, x in rows)
    fails = [t for t in e7["gate"]["tracks"] if not t["finished"]]
    n_out = sum(t["laps"] < 0.5 for t in fails)
    net, base = sm.load_net(d)
    drv = probe.ProbeDriver(net, base)
    rec = robust.laps(drv, seeds=[10000], record=True, outlap=True, **sm.CAR)
    gp = d / "lap-10000-probe.gif"
    render.lap_gif(10000, rec["trajectories"][0], rec["tracks"][0], "probing out-lap + own map (E7)", gp,
                   dirt=dy.dirt_mask(10000))
    gif = base64.b64encode(gp.read_bytes()).decode()
    css = Path("reports/phase35-e2.html").read_text().split("<style>")[1].split("</style>")[0]
    page = f"""<title>Phase 3.5 E7 probes</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}
img.gif {{ width:100%; max-width:480px; border-radius:6px; border:1px solid var(--line) }}
</style>
<main>
<header><div class="eyebrow">Phase 3.5 · E7 · validation set · out-lap + timed lap</div>
<h1>Probing the surface every 6 metres</h1>
<p class="cap">On the out-lap the car gives a 0.4 s full-brake pulse every 6 m when it's going roughly straight, and reads the grip under it from the locked-wheel deceleration (K = {cal['k']:.2f}, fitted on training-range physics; checked on gate-range physics: off by {cal['gate_check']['mean_abs']:.3f} on average). About {e7['probes_per_lap']:.0f} readings per out-lap fill a 5 m map. The timed lap runs on that map alone.</p></header>

<section class="note"><h2>Result</h2>
<p><b>{e7['gate']['completion']:.0%}</b> of gate-range attempts finished (E5 55%, E6 39%; true grip and map 86%). <b>{e7['nominal']['mean_lap_time']:.1f} s</b> at nominal, all 20 finished: inside the 28.7 s bar. Its map is off by {e7['map_err']:.2f} on average, against 0.24 in E5.</p>
<p>Still short of 85%. Of the {len(fails)} failures, {n_out} are on the out-lap and {len(fails) - n_out} on the timed lap. The leftover errors are dirt it couldn't probe (in corners, filled from either side) and the position shift between its out-lap line and its timed-lap line.</p></section>

<section><h2>Four ways to know the grip</h2>
<div class="wide"><table><tr><th>inputs</th><th>gate range</th><th>nominal</th><th>timed lap</th></tr>{trs}</table></div>
<p class="cap">Gate range: 100 attempts (5 physics draws per validation track); nominal: 20 tracks. Bar: ≥ 85%, ≤ 28.7 s.</p>
<div class="wide"><table><tr><th>by grip</th>{"".join(f"<th>{b}×</th>" for b in BINS)}</tr>{grs}</table></div></section>

<section><h2>Out-lap with probes, then the timed lap</h2>
<p class="cap">Validation track 10000 (it starts on dirt), nominal physics, real speed. Watch the pedal gauge on the out-lap: short red pulses are the probes. Brown: dirt.</p>
<img class="gif" alt="Probing out-lap GIF" src="data:image/gif;base64,{gif}"></section>
<p class="cap">Code: sdc/phase35/probe.py, e7_report.py.</p>
</main>
"""
    Path("reports/phase35-e7.html").write_text(page)
    print("reports/phase35-e7.html")


if __name__ == "__main__":
    main()
