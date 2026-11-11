"""Phase 3.5 E4 report: python -m sdc.phase35.e4_report -> reports/phase35-e4.html"""

import base64
import json
from pathlib import Path

import torch

from sdc.phase3 import robust
from sdc.phase3 import train as p3
from sdc.phase35 import e1
from sdc.phase35.e3_report import policy

RUNS = {"hi": "high noise (std 0.5 → 0.1, as E3)", "lo": "low noise (std 0.15 → 0.05)"}


def main():
    torch.set_num_threads(1)
    rows = []
    for n, name in RUNS.items():
        for s in range(2):
            r = Path(f"runs/p35-e4-{n}-s{s}")
            sc = json.loads((r / "ckpt_scores.json").read_text())
            bc = json.loads((Path(f"runs/p35-e4-bc-s{s}") / "metrics.json").read_text())
            rel, fast = e1.choose(sc)
            best = fast or rel
            last = sc[-1]
            rows.append((name, s, bc, best, fast is not None, last))
    pol, car = policy(Path("runs/p35-e4-lo-s0-fast"))
    out = Path("runs/p35-e4-lo-s0-fast")
    gifs = p3.gifs(pol, out, "clone + low-noise PPO (E4, seed 0)", robust.laps(pol, **car), **car)
    gif = base64.b64encode((out / gifs[0]["file"]).read_bytes()).decode()

    def cell(r):
        return f"{r['gate_range']:.0%}<br>{r['mean_lap_time']:.1f} s"
    trs = "".join(
        f"<tr><td>{name.split(' (')[0]}, seed {s}</td>"
        f"<td>{bc['gate_laps']['completion']:.0%}<br>{bc['final_eval']['mean_lap_time']:.1f} s</td>"
        f"<td>{cell(best)}{'' if ok else '<br><i>none ≥ 90%; most reliable</i>'}</td><td>{cell(last)}</td></tr>"
        for name, s, bc, best, ok, last in rows)
    css = Path("reports/phase35-e2.html").read_text().split("<style>")[1].split("</style>")[0]
    page = f"""<title>Phase 3.5 E4 clone</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}
:root {{ --s1:var(--hand); --s2:var(--rl); --s3:var(--est); --s4:var(--thr) }}
.ax {{ stroke:var(--line) }} .axlabel {{ fill:var(--muted); font:12px var(--sans) }} .caption {{ color:var(--muted); font-size:.88rem }}
.sw {{ width:10px; height:10px; border-radius:50%; vertical-align:-1px }}
img.gif {{ width:100%; max-width:480px; border-radius:6px; border:1px solid var(--line) }}
td i {{ color:var(--muted); font-size:.8em; font-style:normal }}
</style>
<main>
<header><div class="eyebrow">Phase 3.5 · E4 · validation set · true grip and true map</div>
<h1>Show it fast driving, then let PPO loose</h1>
<p class="cap">Your network first copies the hand-coded driver (behaviour cloning, 25.4 s, same knowledge as E3's car). Then your PPO trains on from there for 1M steps with E3's reward, at two exploration-noise levels. Two seeds each.</p></header>

<section class="note"><h2>Result</h2>
<p><b>Low noise: it stays fast.</b> Seed 0 keeps a 25.8 s checkpoint at 95% finished, then settles near 28 s at about 90%. Seed 1 stays at 23–25 s but never gets reliable (best 72%).</p>
<p><b>High noise (E3's schedule): it drifts back part of the way</b>, to 28–31 s. Seed 0's pick is 30.3 s at 92%; seed 1 peaks at 89%.</p>
<p>From scratch (E3), the same inputs and reward gave 33–34 s. Gate bar: 30.5 s.</p></section>

<section><h2>Clone → after PPO</h2>
<p class="cap">Each cell: share of 100 gate-range validation laps finished, then mean nominal lap. "Pick" is the fastest checkpoint with ≥ 90% finished.</p>
<div class="wide"><table><tr><th>run</th><th>clone only</th><th>pick</th><th>last (1M)</th></tr>{trs}</table></div>
<p>The clone is fast but fragile: it finished 33–62% of the harder laps, because copying never shows it how to recover from mistakes. PPO trades some speed for that robustness. With big exploration noise the trade goes further, since driving at the limit while the noise shakes the controls ends in crashes.</p></section>

{e1.scatter([f"runs/p35-e4-{n}-s{s}" for n in RUNS for s in range(2)]).replace("seed 0", "high, seed 0").replace("seed 1", "high, seed 1").replace("seed 2", "low, seed 0").replace("seed 3", "low, seed 1")}

<section><h2>One lap</h2>
<p class="cap">Low noise, seed 0, its pick (51k PPO steps after cloning). Validation track 10000, nominal physics, real speed. Brown: dirt, which this car knows about from the map.</p>
<img class="gif" alt="Lap GIF of the E4 car" src="data:image/gif;base64,{gif}"></section>
<p class="cap">This car still uses the true grip and the true map (oracle inputs), so it can't run the gate yet. Code: sdc/phase35/bc.py, e4_report.py; my_ppo --set init=…</p>
</main>
"""
    Path("reports/phase35-e4.html").write_text(page)
    print("reports/phase35-e4.html")


if __name__ == "__main__":
    main()
