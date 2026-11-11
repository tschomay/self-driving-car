"""Phase 3.5 E6 report: python -m sdc.phase35.e6_report -> reports/phase35-e6.html"""

import base64
import json
from collections import defaultdict
from pathlib import Path

import torch

from sdc.phase3 import robust
from sdc.phase35 import braketest as bt
from sdc.phase35 import selfmap as sm
from sdc.phase35.e5_report import BINS, grip_bin
from sdc.racer import dynamics as dy
from sdc.racer import render


def by_grip(tracks):
    by = defaultdict(list)
    for t in tracks:
        by[grip_bin(t["physics"]["grip"])].append(t["finished"])
    return "".join(f"<td>{sum(by[b])}/{len(by[b])}</td>" for b in BINS)


def main():
    torch.set_num_threads(1)
    e5 = json.loads(Path("runs/p35-e5-lo-s0/eval.json").read_text())
    e6 = json.loads(Path("runs/p35-e5-lo-s0/eval_braketest.json").read_text())
    cal = json.loads(bt.CAL.read_text())
    rows = [("true grip and map", e5["oracle"]), ("E5: own estimate and map", e5["self"]), ("E6: brake test + map", e6)]
    trs = "".join(f"<tr><td>{n}</td><td>{x['gate']['completion']:.0%}</td><td>{x['nominal']['completion']:.0%}</td>"
                  f"<td>{x['nominal']['mean_lap_time']:.1f} s</td></tr>" for n, x in rows)
    grs = "".join(f"<tr><td>{n}</td>{by_grip(x['gate']['tracks'])}</tr>" for n, x in rows)
    net, base, est = sm.load("runs/p35-e5-lo-s0")
    drv = bt.BrakeTestDriver(net, base, est)
    rec = robust.laps(drv, seeds=[10003], record=True, outlap=True, **sm.CAR)
    gp = Path("runs/p35-e5-lo-s0/lap-10003-braketest.gif")
    render.lap_gif(10003, rec["trajectories"][0], rec["tracks"][0], "brake test + own map (E6)", gp,
                   dirt=dy.dirt_mask(10003))
    gif = base64.b64encode(gp.read_bytes()).decode()
    css = Path("reports/phase35-e2.html").read_text().split("<style>")[1].split("</style>")[0]
    page = f"""<title>Phase 3.5 E6 brake test</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}
img.gif {{ width:100%; max-width:480px; border-radius:6px; border:1px solid var(--line) }}
</style>
<main>
<header><div class="eyebrow">Phase 3.5 · E6 · validation set · out-lap + timed lap</div>
<h1>Does a brake test fix the grip estimate?</h1>
<p class="cap">At the start of the out-lap the car accelerates to 8 m/s and brakes with locked wheels; the accelerometer gives the grip level (calibrated on training-range physics only, K = {cal['k']:.2f}). The window estimator then only marks where grip changes. Same car and map as E5 (low noise).</p></header>

<section class="note"><h2>Result</h2>
<p><b>Worse.</b> {e6['gate']['completion']:.0%} of gate-range attempts finished ({e5['self']['gate']['completion']:.0%} in E5), {e6['nominal']['completion']:.0%} at nominal. Faster ({e6['nominal']['mean_lap_time']:.1f} s), and the slippery tracks got better, but it now crashes on normal ones.</p>
<p>The brake test reads the grip level well. The map can't see the dirt: in its own map the car rates dirt stretches at 0.945 of the grip level and clean ones at 0.949, so it flags 2% of the dirt. In E5 its overall underestimate hid that by making it cautious everywhere.</p></section>

<section><h2>Three ways to know the grip</h2>
<div class="wide"><table><tr><th>inputs</th><th>gate range</th><th>nominal</th><th>timed lap</th></tr>{trs}</table></div>
<p class="cap">Gate range: 100 attempts (5 physics draws per validation track); nominal: 20 tracks. Bar: ≥ 85%, ≤ 28.7 s.</p>
<div class="wide"><table><tr><th>by grip</th>{"".join(f"<th>{b}×</th>" for b in BINS)}</tr>{grs}</table></div></section>

<section class="note"><h2>Why the map is blind to dirt</h2>
<p>On the out-lap, the car believes everything ahead might be dirt, so it drives gently. Gentle driving barely loads the tires, and a tire that isn't working hard feels the same on dirt as on tarmac. That's E2's lesson again: you only feel grip near the limit. One brake test reads one surface, the one at the start.</p>
<p>Your prediction named the fix: test the surface more than once, so it sees all of them.</p></section>

<section><h2>A failed attempt</h2>
<p class="cap">Validation track 10003, nominal physics, real speed: out-lap with the brake test at the start (pedal gauge), then the timed lap. Brown: dirt, which this car doesn't know about.</p>
<img class="gif" alt="Brake test lap GIF" src="data:image/gif;base64,{gif}"></section>
<p class="cap">Code: sdc/phase35/braketest.py, e6_report.py.</p>
</main>
"""
    Path("reports/phase35-e6.html").write_text(page)
    print("reports/phase35-e6.html")


if __name__ == "__main__":
    main()
