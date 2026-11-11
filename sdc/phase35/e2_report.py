"""Phase 3.5 E2 report page: python -m sdc.phase35.e2_report  ->  reports/phase35-e2.html"""

import json
from pathlib import Path

D = Path("runs/p35-e2")
full = json.loads((D / "full.json").read_text())["best"]
blind = json.loads((D / "blind.json").read_text())
est_rl = json.loads((D / "blind_est-p35-e1-s1-reliable-est.json").read_text())
est_own = json.loads((D / "blind_est-p35-e2-est.json").read_text())
HAND = full["mean_lap_time"]
BAR = 1.2 * HAND

ladder = [
    ("Hand-coded, knows grip, mass and dirt", HAND, full["gate_range"], "hand"),
    ("Hand-coded, no dirt or mass, true grip", blind["best"]["mean_lap_time"], blind["best"]["gate_range"], "hand"),
    ("Hand-coded, no dirt or mass, estimated grip", est_own["best"]["mean_lap_time"], est_own["best"]["gate_range"], "hand"),
    ("RL car (E1, seeds 1–2, most reliable)", 34.5, 0.96, "rl"),
    ("RL car (E1, seed 0, fastest with ≥ 90%)", 36.7, 0.93, "rl"),
]


def bars():
    W, L, R, rowh, T = 640, 300, 80, 34, 22
    H = T + rowh * len(ladder) + 30
    lo, hi = 20, 40
    sx = lambda v: L + (W - L - R) * (v - lo) / (hi - lo)  # noqa: E731
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="lap time by driver">']
    for t in range(lo, hi + 1, 5):
        p.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{T - 6}" y2="{H - 26}" class="grid"/>')
        p.append(f'<text x="{sx(t):.1f}" y="{H - 10}" class="tick" text-anchor="middle">{t} s</text>')
    for i, (name, v, g, kind) in enumerate(ladder):
        y = T + i * rowh
        p.append(f'<text x="{L - 10}" y="{y + 15}" class="lab" text-anchor="end">{name}</text>')
        p.append(f'<rect x="{sx(lo):.1f}" y="{y + 3}" width="{sx(v) - sx(lo):.1f}" height="18" rx="2" class="b-{kind}"/>')
        p.append(f'<text x="{sx(v) + 6:.1f}" y="{y + 16}" class="val">{v:.1f} s · {g:.0%}</text>')
    p.append(f'<line x1="{sx(BAR):.1f}" x2="{sx(BAR):.1f}" y1="{T - 10}" y2="{H - 26}" class="thr"/>')
    p.append(f'<text x="{sx(BAR) + 4:.1f}" y="{T - 10}" class="thrlab">gate 1.2× = {BAR:.1f} s</text>')
    p.append("</svg>")
    return "".join(p)


def frontier():
    W, H, L, R, T, B = 640, 300, 48, 16, 16, 44
    lo, hi = 20, 40
    sx = lambda v: L + (W - L - R) * (v - lo) / (hi - lo)  # noqa: E731
    sy = lambda g: T + (H - T - B) * (1 - g)  # noqa: E731
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="reliability vs lap time for each margin setting">']
    for g in (0, .25, .5, .75, 1):
        p.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(g):.1f}" y2="{sy(g):.1f}" class="grid"/>')
        p.append(f'<text x="{L - 6}" y="{sy(g) + 4:.1f}" class="tick" text-anchor="end">{g:.0%}</text>')
    for t in range(lo, hi + 1, 5):
        p.append(f'<text x="{sx(t):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{t} s</text>')
    p.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(.9):.1f}" y2="{sy(.9):.1f}" class="thr"/>')
    p.append(f'<line x1="{sx(BAR):.1f}" x2="{sx(BAR):.1f}" y1="{T}" y2="{H - B}" class="thr"/>')
    p.append(f'<text x="{(L + W - R) / 2}" y="{H - 6}" class="tick" text-anchor="middle">mean lap at nominal (validation)</text>')
    for res, cls in ((blind, "s1"), (est_own, "s2"), (est_rl, "s3")):
        pts = sorted((r["mean_lap_time"], r["gate_range"]) for r in res["rows"] if r["mean_lap_time"])
        for x, g in pts:
            p.append(f'<circle cx="{sx(x):.1f}" cy="{sy(g):.1f}" r="4" class="{cls}"/>')
    p.append(f'<circle cx="{sx(34.5):.1f}" cy="{sy(.96):.1f}" r="6" class="rl"/>')
    p.append(f'<circle cx="{sx(36.7):.1f}" cy="{sy(.93):.1f}" r="6" class="rl"/>')
    p.append(f'<text x="{W - R - 4}" y="{sy(.9) - 6:.1f}" class="thrlab" text-anchor="end">90%</text>')
    p.append("</svg>")
    return "".join(p)


def tbl(res):
    def lap(r):
        return f"{r['mean_lap_time']:.1f} s" if r["mean_lap_time"] else "–"
    return "".join(f"<tr><td>{r['lat_margin']}</td><td>{r['brake_margin']}</td><td>{r['gate_range']:.0%}</td>"
                   f"<td>{lap(r)}</td></tr>" for r in res["rows"])


page = f"""<title>Phase 3.5 E2 ceiling</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column, lab-notebook style; charts full column width. */
:root {{ --bg:#f6f7f8; --fg:#1b2026; --muted:#5d6672; --line:#d5dae0; --hand:#3a6ea5; --rl:#c4622d; --est:#7a5aa8; --thr:#c0392b; --card:#ffffff;
  --sans:"IBM Plex Sans",system-ui,sans-serif; --mono:"IBM Plex Mono",ui-monospace,monospace }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#14171b; --fg:#e4e8ec; --muted:#9aa4b0; --line:#2c333b; --hand:#6fa0d8; --rl:#e88a55; --est:#a98ad6; --thr:#ef6b5c; --card:#1b1f24; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#14171b; --fg:#e4e8ec; --muted:#9aa4b0; --line:#2c333b; --hand:#6fa0d8; --rl:#e88a55; --est:#a98ad6; --thr:#ef6b5c; --card:#1b1f24; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.55 var(--sans); padding-left:max(16px,env(safe-area-inset-left,0px)); padding-right:max(16px,env(safe-area-inset-right,0px)); padding-block:24px 48px }}
main {{ max-width:720px; margin:0 auto; display:flex; flex-direction:column; gap:28px }}
h1 {{ font-size:1.6rem; margin:0; text-wrap:balance }} h2 {{ font-size:1.1rem; margin:0 0 8px }}
.eyebrow {{ font:500 .75rem var(--mono); letter-spacing:.08em; text-transform:uppercase; color:var(--muted) }}
p {{ margin:0 0 8px; max-width:65ch }} .cap {{ color:var(--muted); font-size:.88rem }}
svg {{ width:100%; height:auto; display:block }} .wide {{ overflow-x:auto }}
.grid {{ stroke:var(--line) }} .tick,.val {{ fill:var(--muted); font:12px var(--mono) }} .lab {{ fill:var(--fg); font:12px var(--sans) }}
.thr {{ stroke:var(--thr); stroke-dasharray:5 4; stroke-width:1.5 }} .thrlab {{ fill:var(--thr); font:12px var(--mono) }}
.b-hand {{ fill:var(--hand) }} .b-rl {{ fill:var(--rl) }}
.s1 {{ fill:var(--hand); fill-opacity:.85 }} .s2 {{ fill:var(--est); fill-opacity:.85 }} .s3 {{ fill:none; stroke:var(--est); stroke-width:1.5 }} .rl {{ fill:var(--rl) }}
.key {{ display:flex; flex-wrap:wrap; gap:6px 16px; font-size:.85rem; color:var(--muted) }} .key i {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:-1px }}
.note {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:14px 16px }}
table {{ border-collapse:collapse; font:13px var(--mono); font-variant-numeric:tabular-nums; width:100% }} td,th {{ border-bottom:1px solid var(--line); padding:4px 8px; text-align:right }} th {{ color:var(--muted); font-weight:500 }}
.tables {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px }} .tables > div {{ min-width:0 }}
</style>
<main>
<header><div class="eyebrow">Phase 3.5 · E2 · validation set</div>
<h1>Is about 33 s the best possible with what the car knows?</h1>
<p class="cap">The hand-coded driver, given only what the RL car knows, with its safety margins tuned to the fastest setting that still finishes ≥ 90% of 100 gate-range laps.</p></header>

<section><h2>Lap time as the driver loses information</h2>
<p class="cap">Mean lap at nominal physics on the 20 validation tracks, and the share of 100 gate-range laps finished. Blue: hand-coded driver. Orange: the RL car from E1. Red dashed line: the gate's speed bar.</p>
<div class="wide">{bars()}</div></section>

<section class="note"><h2>What it says</h2>
<p>Dropping the dirt map and the mass costs the hand-coded driver about {blind['best']['mean_lap_time'] - HAND:.1f} s. Swapping the true grip for an estimate costs another {est_own['best']['mean_lap_time'] - blind['best']['mean_lap_time']:.1f} s. With the RL car's information it laps in {est_own['best']['mean_lap_time']:.1f} s, about the same as the RL car.</p>
<p>So the RL car isn't leaving much on the table. The constraints set the ceiling, and the biggest cost is not knowing the grip well enough.</p></section>

<section><h2>Every margin setting: reliable vs. fast</h2>
<div class="key"><span><i style="background:var(--hand)"></i>true grip, no dirt map</span><span><i style="background:var(--est)"></i>estimate trained on its own driving</span><span><i style="border:1.5px solid var(--est)"></i>estimate trained on the RL car's driving</span><span><i style="background:var(--rl)"></i>RL car (E1)</span></div>
<div class="wide">{frontier()}</div>
<p class="cap">Each dot is one pair of safety margins (cornering 0.4–0.8 of the grip limit, braking 0.3–0.5). Up and left is better. Red lines: 90% finished and the 30.5 s bar. Nothing estimate-driven reaches the top-left corner.</p></section>

<section class="note"><h2>A surprise, checked</h2>
<p>The first run reused the RL car's grip estimator. It made the hand-coded driver slower and less reliable than the RL car (best: 86% finished, 38.4 s). Driving the hand-coded car, that estimator was off by 0.21 on average and read 15% too low, against 0.05 on the RL car's own driving. It had only learned to read grip from the RL car's style.</p>
<p>A new estimator trained on the hand-coded driver's own driving does better (34.8 s at 91%), but it's still off by about 0.17. The hand-coded driver never pushes the tires near their limit, so its driving shows little about the grip. The RL car slides more, and that slide is what the estimator reads.</p></section>

<section><h2>All settings</h2><div class="tables">
<div><p class="cap">True grip, no dirt map</p><table><tr><th>corner</th><th>brake</th><th>finished</th><th>lap</th></tr>{tbl(blind)}</table></div>
<div><p class="cap">Estimated grip (own estimator)</p><table><tr><th>corner</th><th>brake</th><th>finished</th><th>lap</th></tr>{tbl(est_own)}</table></div>
</div></section>
<p class="cap">Caveat: the hand-coded driver still knows the track's shape ahead and reads speed and position without noise, so its numbers are a slightly generous ceiling. Code: sdc/phase35/e2.py, e2_report.py.</p>
</main>
"""
Path("reports").mkdir(exist_ok=True)
Path("reports/phase35-e2.html").write_text(page)
print("reports/phase35-e2.html")
