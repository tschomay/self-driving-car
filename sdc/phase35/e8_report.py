"""Phase 3.5 E8 report: python -m sdc.phase35.e8_report -> reports/phase35-e8.html"""

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
    e8 = json.loads((d / "eval_probe_corners.json").read_text())
    cal = json.loads(probe.CAL_CORNERS.read_text())
    rows = [("true grip and map", e5["oracle"]), ("E5: window estimate", e5["self"]), ("E6: one brake test", e6),
            ("E7: probes every 6 m", e7), ("E8: probes in corners too", e8)]
    trs = "".join(f"<tr><td>{n}</td><td>{x['gate']['completion']:.0%}</td><td>{x['nominal']['completion']:.0%}</td>"
                  f"<td>{x['nominal']['mean_lap_time']:.1f} s</td></tr>" for n, x in rows)
    grs = "".join(f"<tr><td>{n}</td>{by_grip(x['gate']['tracks'])}</tr>" for n, x in rows)
    fails = [t for t in e8["gate"]["tracks"] if not t["finished"]]
    n_out = sum(t["laps"] < 0.5 for t in fails)
    net, base = sm.load_net(d)
    drv = probe.ProbeDriver(net, base, corners=True)
    rec = robust.laps(drv, seeds=[10000], record=True, outlap=True, **sm.CAR)
    gp = d / "lap-10000-probe-corners.gif"
    render.lap_gif(10000, rec["trajectories"][0], rec["tracks"][0], "corner probes + own map (E8)", gp,
                   dirt=dy.dirt_mask(10000))
    gif = base64.b64encode(gp.read_bytes()).decode()
    css = Path("reports/phase35-e2.html").read_text().split("<style>")[1].split("</style>")[0]
    page = f"""<title>Phase 3.5 E8 corner probes</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}
img.gif {{ width:100%; max-width:480px; border-radius:6px; border:1px solid var(--line) }}
</style>
<main>
<header><div class="eyebrow">Phase 3.5 · E8 · validation set · out-lap + timed lap</div>
<h1>Probing in corners too</h1>
<p class="cap">E7's probes, now allowed mid-corner. A probe reads the total horizontal acceleration (braking and cornering), since a locked, sliding tire pushes with the same total force in whatever direction it slides (K = {cal['k']:.2f}, checked on gate-range physics: off by {cal['gate_check']['mean_abs']:.3f}).</p></header>

<section class="note"><h2>Result</h2>
<p><b>No change: {e8['gate']['completion']:.0%}</b> of gate-range attempts finished (E7: {e7['gate']['completion']:.0%}), {e8['nominal']['mean_lap_time']:.1f} s at nominal. Of {len(fails)} failures, {n_out} are on the out-lap and {len(fails) - n_out} on the timed lap.</p>
<p>Checked why: corner probes barely happened. On the 20 nominal tracks the car made 611 probes against 600 in E7, and the share taken while turning (gyro above 0.1) went from 49% to 50%. A probe needs at least 6 m/s for a clean reading, and on its cautious out-lap the car corners slower than that. The throttle override that brings it up to speed only works when it's going straight, so it doesn't help in corners. The steering rule wasn't what blocked corner probes; speed was.</p></section>

<section><h2>Five ways to know the grip</h2>
<div class="wide"><table><tr><th>inputs</th><th>gate range</th><th>nominal</th><th>timed lap</th></tr>{trs}</table></div>
<p class="cap">Gate range: 100 attempts (5 physics draws per validation track); nominal: 20 tracks. Bar: ≥ 85%, ≤ 28.7 s.</p>
<div class="wide"><table><tr><th>by grip</th>{"".join(f"<th>{b}×</th>" for b in BINS)}</tr>{grs}</table></div></section>

<section><h2>Out-lap with probes, then the timed lap</h2>
<p class="cap">Validation track 10000 (it starts on dirt), nominal physics, real speed. Short red pedal pulses on the out-lap are the probes. Brown: dirt.</p>
<img class="gif" alt="Corner probe lap GIF" src="data:image/gif;base64,{gif}"></section>
<p class="cap">Code: sdc/phase35/probe.py (--corners), e8_report.py.</p>
</main>
"""
    Path("reports/phase35-e8.html").write_text(page)
    print("reports/phase35-e8.html")


if __name__ == "__main__":
    main()
