"""Phase 3.5 E5 report: python -m sdc.phase35.e5_report -> reports/phase35-e5.html"""

import base64
import json
from collections import defaultdict
from pathlib import Path

import torch

from sdc.phase3 import robust
from sdc.phase35 import selfmap as sm
from sdc.racer import dynamics as dy
from sdc.racer import render

NAMES = {"lo": "low noise (E4 25.8 s)", "hi": "high noise (E4 30.3 s)"}
BINS = ("<0.7", "0.7–0.9", "0.9–1.1", ">1.1")


def grip_bin(g):
    return BINS[0] if g < 0.7 else BINS[1] if g < 0.9 else BINS[2] if g < 1.1 else BINS[3]


def main():
    torch.set_num_threads(1)
    hand = robust.laps(dy.DynamicPursuit(), outlap=True, **sm.CAR)["mean_lap_time"]
    bar = 1.2 * hand
    rows, grip_rows = [], []
    for n, name in NAMES.items():
        r = json.loads((Path(f"runs/p35-e5-{n}-s0") / "eval.json").read_text())
        for k, lab in (("oracle", "true grip and map"), ("self", "own estimate and map")):
            x = r[k]
            t = x["nominal"]["mean_lap_time"]
            rows.append(f"<tr><td>{name.split(' (')[0]}</td><td>{lab}</td><td>{x['gate']['completion']:.0%}</td>"
                        f"<td>{x['nominal']['completion']:.0%}</td><td>{t:.1f} s ({t / hand:.2f}×)</td></tr>")
            by = defaultdict(list)
            for tr in x["gate"]["tracks"]:
                by[grip_bin(tr["physics"]["grip"])].append(tr["finished"])
            grip_rows.append(f"<tr><td>{name.split(' (')[0]}, {lab}</td>" + "".join(
                f"<td>{sum(by[b])}/{len(by[b])}</td>" for b in BINS) + "</tr>")
    net, base, est = sm.load("runs/p35-e5-lo-s0")
    drv = sm.SelfMapDriver(net, base, est)
    rec = robust.laps(drv, seeds=[10000], record=True, outlap=True, **sm.CAR)
    gif_path = Path("runs/p35-e5-lo-s0/lap-10000.gif")
    render.lap_gif(10000, rec["trajectories"][0], rec["tracks"][0], "own map + grip estimate (E5, low noise)",
                   gif_path, dirt=dy.dirt_mask(10000))
    gif = base64.b64encode(gif_path.read_bytes()).decode()
    lo = json.loads(Path("runs/p35-e5-lo-s0/eval.json").read_text())
    hi = json.loads(Path("runs/p35-e5-hi-s0/eval.json").read_text())
    css = Path("reports/phase35-e2.html").read_text().split("<style>")[1].split("</style>")[0]
    page = f"""<title>Phase 3.5 E5 self-map</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}
img.gif {{ width:100%; max-width:480px; border-radius:6px; border:1px solid var(--line) }}
</style>
<main>
<header><div class="eyebrow">Phase 3.5 · E5 · validation set · out-lap + timed lap</div>
<h1>Can the car map the track itself?</h1>
<p class="cap">E4's two seed-0 cars, now with no oracle inputs. Grip comes from a 1 s window estimator; position from adding up its own speed readings, reset at the start line. On the out-lap it drives as if everything ahead might be dirt and records the grip it feels in 5 m stretches; on the timed lap it reads that map 5–40 m ahead. Both laps must be finished; only the timed lap counts for time.</p></header>

<section class="note"><h2>Result</h2>
<p><b>Speed holds, reliability doesn't.</b> The low-noise car laps in {lo['self']['nominal']['mean_lap_time']:.1f} s on its own map ({lo['self']['nominal']['mean_lap_time'] / hand:.2f}× the hand-coded driver), inside the gate bar, but finishes only {lo['self']['gate']['completion']:.0%} of gate-range attempts. The high-noise car: {hi['self']['nominal']['mean_lap_time']:.1f} s, {hi['self']['gate']['completion']:.0%}.</p>
<p>Almost all the extra failures are at grip below 0.7×, below anything it trained on.</p></section>

<section><h2>Own map vs. true map</h2>
<p class="cap">Gate range: 100 attempts (5 physics draws per validation track). Nominal: 20 tracks. Lap: the timed lap, a flying start after the out-lap. On this protocol the hand-coded driver laps in {hand:.1f} s, so the gate's 1.2× bar is {bar:.1f} s (it was 30.5 s for a standing start).</p>
<div class="wide"><table><tr><th>car</th><th>inputs</th><th>gate range</th><th>nominal</th><th>lap</th></tr>{"".join(rows)}</table></div></section>

<section><h2>Where it fails: by grip</h2>
<p class="cap">Finished / attempts in the gate-range draws, by the episode's grip. Training covered 0.7–1.1×.</p>
<div class="wide"><table><tr><th>car</th>{"".join(f"<th>{b}×</th>" for b in BINS)}</tr>{"".join(grip_rows)}</table></div>
<p>With the true grip and map, the low-noise car handles grip below 0.7× (24 of 27). With its own estimate it finishes 1 of 27. Its map is off by {lo['self']['map_err']:.2f} on average (the high-noise car's by {hi['self']['map_err']:.2f}). The estimator learned grip from driving within 0.7–1.1×, and its readings pull toward the middle of that range. On a 0.55× track the low-noise car's estimate reads 0.16 too high, so it trusts grip that isn't there. At 1.0× it reads 0.20 too low, which costs it about 2.5 s against the true map. The odometer drifts by about {lo['self']['odo_err_abs']:.1f} m against the centerline. That's less than one 5 m stretch, and it's measured against the centerline, which the car doesn't follow exactly.</p></section>

<section><h2>Out-lap and timed lap</h2>
<p class="cap">Low-noise car on its own map and estimate: validation track 10000, nominal physics, real speed. The first lap is the cautious out-lap; the second is the timed lap on the map it recorded.</p>
<img class="gif" alt="Out-lap and timed lap GIF" src="data:image/gif;base64,{gif}"></section>
<p class="cap">Code: sdc/phase35/selfmap.py, e5_report.py.</p>
</main>
"""
    Path("reports/phase35-e5.html").write_text(page)
    print("reports/phase35-e5.html", hand)


if __name__ == "__main__":
    main()
