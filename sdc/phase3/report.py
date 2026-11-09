"""Phone-readable report page for Phase 3 (dynamic car) runs, built on the Phase 1 report.

Adds: gate-range completion per row, the gate-range curve during training (dashed), and
robustness curves (held-out completion as one physics knob changes, the others nominal).

Usage:
    python -m sdc.phase3.report --title "E1 ..." --question "..." --out reports/p3-e1.html \
        runs/p3-pursuit runs/p3-phase1 runs/p3-e1-s0 ...
"""

from __future__ import annotations

import argparse
import html
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from sdc.phase1 import report as p1
from sdc.phase3 import robust
from sdc.report import N_SERIES, _nice_ticks

GIF_CAPTION = ("Each GIF is one lap attempt on one held-out track at nominal physics, from a standing start, at "
               "real speed (10 steps per second). Brown stretches are dirt (65% grip); the car can't see them. Blue "
               "lines are the 11 sensor rays. The thin trail shows speed (blue slow, orange fast). Dark skid marks "
               "appear where an axle slides more than 4° sideways, darker and thicker up to 20° (tires grip best "
               "around 6–9°; far beyond that the car is sliding). Gauges under the map: steering (blue, left or "
               "right of centre) and pedal (red brake to the left, green gas to the right). An end card says how it ended, then the GIF loops. GIFs show the first seed of a "
               "row; a second GIF, if present, is the first held-out track it failed.")

RESULTS_CAPTION = ("Big number: share of the 20 held-out tracks (never trained on) finished in one lap from a "
                   "standing start at nominal physics, with dirt patches. Below it, 'gate range': 100 laps (5 per "
                   "held-out track) with grip, mass, delay and noise drawn from the gate's range, wider than "
                   "training. Gate: ≥ 85% there. 'Train range': the same with the training range.")

KNOBS = dict(grip=("tire grip (× nominal)", 1.0, "{:g}×"), delay=("control delay (ms)", 1000.0, "{:g}"),
             mass=("mass (× nominal)", 1.0, "{:g}×"), noise=("sensor noise (% of full scale)", 100.0, "{:g}%"))


def _sw(i):
    return f'<span class="sw" style="display:inline-block;background: var(--s{i % N_SERIES + 1})"></span>'


def sweep_chart(series, knob):
    """Completion (%) vs one physics knob. Bands: training range (darker) inside gate range (lighter)."""
    name, scale, fmt = KNOBS[knob]
    W, H, L, R, T, B = 440, 240, 44, 10, 14, 40
    xs = [v * scale for v in robust.SWEEPS[knob]]
    x_lo, x_hi = min(xs), max(xs)
    if knob in ("delay", "noise"):
        x_lo = 0.0

    def sx(x):
        return L + (W - L - R) * (x - x_lo) / (x_hi - x_lo)

    def sy(y):
        return T + (H - T - B) * (1 - y / 105)

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="completion vs {html.escape(name)}">']
    for (lo, hi), op in ((robust.GATE_SPAN[knob], .10), (robust.TRAIN_SPAN[knob], .16)):
        a, b = sx(max(lo * scale, x_lo)), sx(min(hi * scale, x_hi))
        p.append(f'<rect x="{a:.1f}" y="{T}" width="{max(b - a, 2):.1f}" height="{H - T - B}" '
                 f'style="fill: var(--s1)" fill-opacity="{op}"/>')
    for t in (0, 25, 50, 75, 100):
        p.append(f'<line class="ax" x1="{L}" x2="{W - R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke-opacity=".6"/>')
        p.append(f'<text class="tick" x="{L - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{t}</text>')
    for t in _nice_ticks(x_lo, x_hi, 6):
        if x_lo - 1e-9 <= t <= x_hi + 1e-9:
            p.append(f'<text class="tick" x="{sx(t):.1f}" y="{H - B + 16}" text-anchor="middle">{fmt.format(round(t, 3))}</text>')
    p.append(f'<line class="thr" x1="{L}" x2="{W - R}" y1="{sy(85):.1f}" y2="{sy(85):.1f}"/>')
    p.append(f'<text class="axlabel" x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle">{html.escape(name)}</text>')
    p.append(f'<text class="axlabel" x="12" y="{(T + H - B) / 2}" text-anchor="middle" '
             f'transform="rotate(-90 12 {(T + H - B) / 2})">completion (%)</text>')
    for s in series:
        pts = " ".join(f"{sx(x * scale):.1f},{sy(y):.1f}" for x, y in zip(s["x"], s["y"]))
        c = f'var(--s{s["color"] % N_SERIES + 1})'
        p.append(f'<polyline points="{pts}" fill="none" style="stroke: {c}" stroke-width="2" '
                 f'stroke-dasharray="{s.get("dash", "")}" stroke-linejoin="round"/>')
        for x, y in zip(s["x"], s["y"]):
            p.append(f'<circle cx="{sx(x * scale):.1f}" cy="{sy(y):.1f}" r="2.5" style="fill: {c}"/>')
    p.append("</svg>")
    return "".join(p)


def robustness_section(run_dirs) -> str:
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for d in run_dirs:
        m = json.loads((Path(d) / "metrics.json").read_text())
        groups.setdefault(m["label"], []).append(m)
    color = {label: i for i, label in enumerate(groups)}
    charts = []
    for knob in KNOBS:
        series = []
        for label, ms in groups.items():
            have = [m["sweep"][knob] for m in ms if knob in m.get("sweep", {})]
            if not have:
                continue
            x = [r["value"] for r in have[0]]
            y = [100 * float(np.mean([h[i]["completion"] for h in have])) for i in range(len(x))]
            series.append(dict(x=x, y=y, color=color[label], dash="5 3" if ms[0]["preset"] == "pursuit" else ""))
        if series:
            charts.append(f'<div class="chart">{sweep_chart(series, knob)}</div>')
    if not charts:
        return ""
    legend = " · ".join(f"{_sw(color[label])} {html.escape(label)}" for label in groups)
    return ('<section><h2>Robustness curves</h2><p class="caption">Held-out completion (20 tracks, with dirt) as one '
            'physics knob changes and the others stay nominal. Darker band: training range used by randomized runs; '
            'lighter band: the gate\'s wider range. Red line: 85%. Rows with several seeds show the seed average. '
            f'Dashed: the hand-coded driver (it knows the grip and the mass).</p><p class="caption">{legend}</p>'
            f'{"".join(charts)}</section>')


def _extra_sub(ms):
    out = []
    g = [m["gate_laps"]["completion"] for m in ms if "gate_laps" in m]
    t = [m["train_range_laps"]["completion"] for m in ms if "train_range_laps" in m]
    if t:
        out.append(f"train range: {100 * np.mean(t):.0f}% of 100 laps")
    if g:
        out.append(f"gate range: {100 * np.mean(g):.0f}% of 100 laps" + (
            " (per seed: " + ", ".join(f"{100 * x:.0f}%" for x in g) + ")" if len(g) > 1 else ""))
    return out


def _extra_eval(m, color):
    ev = [e for e in m.get("evals", []) if "gate_range" in e]
    if not ev:
        return []
    return [dict(x=[e["timesteps"] for e in ev], y=[100 * e["gate_range"] for e in ev], color=color, width=1.2,
                 opacity=.5, dot=False)]


def build(run_dirs, title, question, out, notes=None, sweep=True, extra_sections=None, **kw):
    extra = ([robustness_section(run_dirs)] if sweep else []) + list(extra_sections or [])
    return p1.build(run_dirs, title, question, out, eyebrow="Phase 3", notes=notes, gif_caption=GIF_CAPTION,
                    footer="sdc/phase3/report.py", extra_sections=[e for e in extra if e],
                    results_caption=RESULTS_CAPTION, base_name="hand-coded driver", extra_sub=_extra_sub,
                    eval_threshold=85, extra_eval=_extra_eval,
                    eval_caption="Thin faint lines: 20 laps with physics drawn from the gate range (one per track).",
                    **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--title", required=True)
    p.add_argument("--question", default="")
    p.add_argument("--note", action="append", default=[])
    p.add_argument("--curve", default="laps", choices=["laps", "return", "speed"])
    p.add_argument("--gifs-per-group", type=int, default=2)
    p.add_argument("--no-sweep", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(build(a.runs, a.title, a.question, a.out, a.note, sweep=not a.no_sweep, curve=a.curve,
                gifs_per_group=a.gifs_per_group))


if __name__ == "__main__":
    main()
