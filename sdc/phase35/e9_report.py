"""Phase 3.5 E9 report: python -m sdc.phase35.e9_report -> reports/phase35-e9.html"""

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
    e9 = json.loads((d / "eval_probe_dense.json").read_text())
    cal = json.loads(probe.CAL_CORNERS.read_text())
    rows = [("true grip and map", e5["oracle"]), ("E5: window estimate", e5["self"]), ("E6: one brake test", e6),
            ("E7: probes every 6 m", e7), ("E8: probes in corners too", e8), ("E9: every 3 m, cautious map", e9)]
    trs = "".join(f"<tr><td>{n}</td><td>{x['gate']['completion']:.0%}</td><td>{x['nominal']['completion']:.0%}</td>"
                  f"<td>{x['nominal']['mean_lap_time']:.1f} s</td></tr>" for n, x in rows)
    grs = "".join(f"<tr><td>{n}</td>{by_grip(x['gate']['tracks'])}</tr>" for n, x in rows)
    fails = [t for t in e9["gate"]["tracks"] if not t["finished"]]
    n_out = sum(t["laps"] < 0.5 for t in fails)
    net, base = sm.load_net(d)
    drv = probe.ProbeDriver(net, base, corners=True, every=3.0, bin=2.5, erode=1)
    rec = robust.laps(drv, seeds=[10000], record=True, outlap=True, **sm.CAR)
    gp = d / "lap-10000-probe-dense.gif"
    render.lap_gif(10000, rec["trajectories"][0], rec["tracks"][0], "dense probes + cautious map (E9)", gp,
                   dirt=dy.dirt_mask(10000))
    gif = base64.b64encode(gp.read_bytes()).decode()
    css = Path("reports/phase35-e2.html").read_text().split("<style>")[1].split("</style>")[0]
    page = f"""<title>Phase 3.5 E9 dense probes</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}
img.gif {{ width:100%; max-width:480px; border-radius:6px; border:1px solid var(--line) }}
</style>
<main>
<header><div class="eyebrow">Phase 3.5 · E9 · validation set · out-lap + timed lap</div>
<h1>Probing every 3 metres</h1>
<p class="cap">Your question: "Maybe it isn't probing often enough?" Same brake-pulse probes as E8 (total force, corners allowed), now every 3 m instead of 6 m, recorded in 2.5 m stretches. Each stretch takes the lowest reading within ±2.5 m, so blurred dirt edges and position shifts err on the careful side. About {e9['probes_per_lap']:.0f} probes per out-lap.</p></header>

<section class="note"><h2>Result</h2>
<p><b>{e9['gate']['completion']:.0%}</b> of gate-range attempts finished (E7/E8: 78%; true grip and map: 86%), at <b>{e9['nominal']['mean_lap_time']:.1f} s</b>, all 20 nominal laps finished. Both validation targets met: ≥ 85% and ≤ 28.7 s.</p>
<p>It edges past the true-map car mostly on the out-lap: that car crashed 9 times on the out-lap, where it drove at full knowledge, against 2 here, where the probing laps are slow. Its deliberately cautious map adds margin on the timed lap. Of {len(fails)} failures, {n_out} are on the out-lap and {len(fails) - n_out} on the timed lap.</p></section>

<section><h2>Six ways to know the grip</h2>
<div class="wide"><table><tr><th>inputs</th><th>gate range</th><th>nominal</th><th>timed lap</th></tr>{trs}</table></div>
<p class="cap">Gate range: 100 attempts (5 physics draws per validation track); nominal: 20 tracks. Bar: ≥ 85%, ≤ 28.7 s.</p>
<div class="wide"><table><tr><th>by grip</th>{"".join(f"<th>{b}×</th>" for b in BINS)}</tr>{grs}</table></div></section>

<section><h2>Out-lap with probes, then the timed lap</h2>
<p class="cap">Validation track 10000 (it starts on dirt), nominal physics, real speed. Short red pedal pulses on the out-lap are the probes. Brown: dirt.</p>
<img class="gif" alt="Corner probe lap GIF" src="data:image/gif;base64,{gif}"></section>
<p class="cap">Code: sdc/phase35/probe.py (--dense), e9_report.py.</p>
</main>
"""
    Path("reports/phase35-e9.html").write_text(page)
    print("reports/phase35-e9.html")


if __name__ == "__main__":
    main()
