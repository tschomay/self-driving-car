"""Phase 3.5 E3 report: python -m sdc.phase35.e3_report -> reports/phase35-e3.html

Checkpoint scatter (as E1), the map ablation (true map / map with no dirt / map that's all dirt) on each seed's
fastest-with-90% pick, and a lap GIF of seed 1's pick.
"""

import base64
import json
from pathlib import Path

import numpy as np
import torch

from sdc.phase3 import my_ppo, robust
from sdc.phase3 import train as p3
from sdc.phase35 import e1
from sdc.racer import dynamics as dy

RUNS = [Path(f"runs/p35-e3-s{s}") for s in range(3)]
OFF = dy.OBS_DIM + 3   # gripmap columns, after accel (2) and grip (1)


def policy(d, mode="true"):
    m = json.loads((d / "metrics.json").read_text())
    car = p3.car_of(m["env"])
    net = my_ppo.owner.ActorCritic(obs_dim=dy.DynamicRacerBatch(1, **car).obs_dim)
    net.load_state_dict(torch.load(d / "model.pt"))

    def act(b):
        o = b.obs()
        if mode == "flat":
            o[:, OFF:OFF + 8] = b.grip[:, None]
        elif mode == "alldirt":
            o[:, OFF:OFF + 8] = b.grip[:, None] * dy.DIRT_GRIP
        with torch.no_grad():
            return np.tanh(net.actor(torch.as_tensor(o)).numpy())
    return act, car


def main():
    torch.set_num_threads(1)
    rows = []
    for r in RUNS:
        d = Path(f"{r}-fast")
        pick = json.loads((d / "pick.json").read_text())
        res = {}
        for mode in ("true", "flat", "alldirt"):
            pol, car = policy(d, mode)
            ev = robust.laps(pol, **car)
            res[mode] = (ev["completion"], ev["mean_lap_time"])
        rows.append((r.name[-2:], pick, res))
    pol, car = policy(Path("runs/p35-e3-s1-fast"))
    out = Path("runs/p35-e3-s1-fast")
    final = robust.laps(pol, **car)
    gifs = p3.gifs(pol, out, "RL car + perfect grip map (E3, seed 1)", final, **car)
    gif = base64.b64encode((out / gifs[0]["file"]).read_bytes()).decode()

    def lap(t):
        return f"{t:.1f} s" if t else "–"
    trs = "".join(
        f"<tr><td>{s} ({p['timesteps'] // 1000}k)</td>"
        f"<td>{res['true'][0]:.0%}<br>{lap(res['true'][1])}</td><td>{res['flat'][0]:.0%}<br>{lap(res['flat'][1])}</td>"
        f"<td>{res['alldirt'][0]:.0%}<br>{lap(res['alldirt'][1])}</td></tr>" for s, p, res in rows)
    css = Path("reports/phase35-e2.html").read_text().split("<style>")[1].split("</style>")[0]
    page = f"""<title>Phase 3.5 E3 map</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}
/* E1's scatter uses series tokens */
:root {{ --s1:var(--hand); --s2:var(--rl); --s3:var(--est) }}
.ax {{ stroke:var(--line) }} .axlabel {{ fill:var(--muted); font:12px var(--sans) }} .caption {{ color:var(--muted); font-size:.88rem }}
.sw {{ width:10px; height:10px; border-radius:50%; vertical-align:-1px }}
img.gif {{ width:100%; max-width:480px; border-radius:6px; border:1px solid var(--line) }}
</style>
<main>
<header><div class="eyebrow">Phase 3.5 · E3 · validation set · true grip and true map</div>
<h1>What is a perfect grip map worth to the RL car?</h1>
<p class="cap">Your PPO, E1's recipe plus the true grip at 8 points 5–40 m ahead (dirt included), from the car's exact position. Three seeds, 1M steps. This is an upper limit: no map the car builds itself can be better.</p></header>

<section class="note"><h2>Result</h2>
<p>Fastest checkpoint with ≥ 90% finished: <b>32.8, 33.7, 34.2 s</b> (seeds 0–2). Without the map (E1, true grip under the car): 34.1 s. The hand-coded driver with the same knowledge: 25.4 s. Gate bar: 30.5 s.</p>
<p>The perfect map is worth about 1 s to the RL car, against 4.5 s to the hand-coded driver.</p></section>

{e1.scatter([str(r) for r in RUNS])}

<section><h2>Does it use the map? Lie to it and see</h2>
<p class="cap">Each seed's pick on the 20 validation tracks at nominal physics, real dirt on the ground. "No dirt" tells the car the track is all clean; "all dirt" tells it everything ahead is dirt. Cells: laps finished, then mean lap.</p>
<div class="wide"><table><tr><th>seed</th><th>true map</th><th>“no dirt”</th><th>“all dirt”</th></tr>{trs}</table></div>
<p>It reads the map. Told there's no dirt it goes off the road far more often; told it's all dirt it slows by up to 10 s. But even when it believes the whole track is clean, it laps in 31.5–34 s. It uses the map to slow down for dirt, not to go faster on clean track. Its pace on clean track is what's slow.</p></section>

<section><h2>One lap</h2>
<p class="cap">Seed 1's pick, validation track 10000, nominal physics, real speed. Brown stretches are dirt (the car now knows where they are). Gauges: steering and pedal.</p>
<img class="gif" alt="Lap GIF of the E3 car" src="data:image/gif;base64,{gif}"></section>

<section class="note"><h2>Checked along the way</h2>
<p>Is it the rays' short view? The hand-coded driver needs only 20 m of lookahead to lap in 25.4 s (95% finished; 30 m or more: 97%). The rays reach 40 m, so the car can see far enough.</p>
<p>That revises E2's reading. With the grip estimate, the RL car matches the hand-coded driver (34.5 vs 34.8 s). Give both better information and only the hand-coded driver gets faster: 29.9 s with the true grip, 25.4 s with the map, while the RL car stays at 33–34 s. The limit is in what the RL car learns.</p></section>
<p class="cap">Code: sdc/phase35/e3_report.py; sensor: sdc/racer/dynamics.py ("gripmap").</p>
</main>
"""
    Path("reports/phase35-e3.html").write_text(page)
    print("reports/phase35-e3.html")


if __name__ == "__main__":
    main()
