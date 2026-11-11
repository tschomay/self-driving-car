"""Phase 3.5 E1: pick checkpoints for speed, not just safety.

    python -m sdc.phase35.e1 score --run runs/p35-e1-s0      # nominal lap time of every checkpoint
    python -m sdc.phase35.e1 pick  --run runs/p35-e1-s0      # writes runs/p35-e1-s0-{reliable,fast}/ policy dirs
    python -m sdc.phase35.e1 report --runs runs/p35-e1-s0 runs/p35-e1-s1 runs/p35-e1-s2

A training run with select=true saves a checkpoint every 50k steps and scores each on the validation gate
range (100 laps, true grip). `score` adds each checkpoint's nominal lap (20 validation tracks, true grip).
`pick` makes two policy directories: "reliable" (Phase 3's rule: best gate range, latest on ties) and "fast"
(fastest nominal mean lap among checkpoints with gate range >= MIN_GATE). Each then gets a 1 s window grip
estimator (sdc.phase3.grip_estimator --kind window), so both are judged driving on the estimate.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from sdc.phase3 import my_ppo, report, robust
from sdc.phase3 import train as p3
from sdc.report import N_SERIES
from sdc.racer import dynamics as dy

MIN_GATE = 0.90


def _greedy(net, squash):
    def act(b):
        with torch.no_grad():
            a = net.actor(torch.as_tensor(b.obs())).numpy()
        return np.tanh(a) if squash else a
    return act


def score(run):
    run = Path(run)
    m = json.loads((run / "metrics.json").read_text())
    car = p3.car_of(m["env"])
    probe = dy.DynamicRacerBatch(1, **car)
    net = my_ppo.owner.ActorCritic(obs_dim=probe.obs_dim)
    pol = _greedy(net, m["env"]["squash"])
    rows = []
    for c in m["selection"]["checkpoints"]:
        net.load_state_dict(torch.load(run / "ckpt" / f"{c['timesteps']}.pt"))
        ev = robust.laps(pol, **car)
        rows.append(dict(c, completion=ev["completion"], mean_lap_time=ev["mean_lap_time"]))
        print(rows[-1], flush=True)
    (run / "ckpt_scores.json").write_text(json.dumps(rows, indent=1))
    return rows


def choose(rows):
    reliable = max(rows, key=lambda r: (r["gate_range"], r["timesteps"]))
    ok = [r for r in rows if r["gate_range"] >= MIN_GATE and r["mean_lap_time"]]
    fast = min(ok, key=lambda r: r["mean_lap_time"]) if ok else None
    return reliable, fast


def pick(run):
    run = Path(run)
    rows = json.loads((run / "ckpt_scores.json").read_text())
    picks = dict(zip(("reliable", "fast"), choose(rows)))
    for name, r in picks.items():
        if r is None:
            print(f"{name}: no checkpoint reaches {MIN_GATE:.0%}")
            continue
        d = Path(f"{run}-{name}")
        d.mkdir(exist_ok=True)
        shutil.copy(run / "ckpt" / f"{r['timesteps']}.pt", d / "model.pt")
        shutil.copy(run / "metrics.json", d / "metrics.json")
        (d / "pick.json").write_text(json.dumps(dict(r, rule=name), indent=1))
        print(name, r)


def scatter(runs):
    """Per checkpoint: validation gate range (y) vs nominal lap (x), one colour per seed; picks ringed."""
    W, H, L, R, T, B = 440, 280, 44, 10, 14, 40
    all_rows = [json.loads((Path(r) / "ckpt_scores.json").read_text()) for r in runs]
    xs = [r["mean_lap_time"] for rows in all_rows for r in rows if r["mean_lap_time"]]
    x_lo, x_hi = np.floor(min(xs + [25.4]) - 1), np.ceil(max(xs) + 1)

    def sx(x):
        return L + (W - L - R) * (x - x_lo) / (x_hi - x_lo)

    def sy(y):
        return T + (H - T - B) * (1 - y / 105)

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="reliability vs lap time per checkpoint">']
    for t in (0, 25, 50, 75, 100):
        p.append(f'<line class="ax" x1="{L}" x2="{W - R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke-opacity=".6"/>')
        p.append(f'<text class="tick" x="{L - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{t}</text>')
    for t in report._nice_ticks(x_lo, x_hi, 6):
        if x_lo <= t <= x_hi:
            p.append(f'<text class="tick" x="{sx(t):.1f}" y="{H - B + 16}" text-anchor="middle">{t:g}</text>')
    p.append(f'<line class="thr" x1="{L}" x2="{W - R}" y1="{sy(90):.1f}" y2="{sy(90):.1f}"/>')
    for x, lab in ((25.4, "hand-coded"), (25.4 * 1.2, "1.2×")):
        p.append(f'<line class="ax" x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{H - B}" stroke-dasharray="4 3"/>')
        p.append(f'<text class="tick" x="{sx(x) + 3:.1f}" y="{T + 10}">{lab}</text>')
    p.append(f'<text class="axlabel" x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle">mean lap at nominal (s)</text>')
    p.append(f'<text class="axlabel" x="12" y="{(T + H - B) / 2}" text-anchor="middle" '
             f'transform="rotate(-90 12 {(T + H - B) / 2})">gate range (%)</text>')
    for i, rows in enumerate(all_rows):
        c = f'var(--s{i % N_SERIES + 1})'
        rel, fast = choose(rows)
        for r in rows:
            if not r["mean_lap_time"]:
                continue
            p.append(f'<circle cx="{sx(r["mean_lap_time"]):.1f}" cy="{sy(100 * r["gate_range"]):.1f}" r="3" '
                     f'style="fill: {c}" fill-opacity=".7"><title>{r["timesteps"] // 1000}k</title></circle>')
        for q, shape in ((rel, "rel"), (fast, "fast")):
            if q:
                cx, cy = sx(q["mean_lap_time"]), sy(100 * q["gate_range"])
                if shape == "rel":
                    p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="none" style="stroke: {c}" stroke-width="2"/>')
                else:
                    p.append(f'<rect x="{cx - 7:.1f}" y="{cy - 7:.1f}" width="14" height="14" fill="none" '
                             f'style="stroke: {c}" stroke-width="2"/>')
    p.append("</svg>")
    legend = " · ".join(f"{report._sw(i)} seed {i}" for i in range(len(runs)))
    return ('<section><h2>Every checkpoint: reliable vs. fast</h2><p class="caption">Each dot is one saved checkpoint '
            '(every 50k steps, true grip). Up: share of 100 validation laps finished with gate-range physics. Left: '
            'faster mean lap at nominal (20 validation tracks). Circle: the most reliable checkpoint (Phase 3\'s '
            'rule). Square: the fastest with ≥ 90% (red line). Dashed: the hand-coded driver (25.4 s) and 1.2× of '
            f'it, the gate\'s speed bar.</p><p class="caption">{legend}</p><div class="chart">{"".join(p)}</div></section>')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["score", "pick", "report"])
    ap.add_argument("--run")
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--est", nargs="*", help="estimator run dirs for the report")
    ap.add_argument("--notes", nargs="*", default=[])
    ap.add_argument("--out", default="reports/phase35-e1.html")
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.cmd == "score":
        score(a.run)
    elif a.cmd == "pick":
        pick(a.run)
    else:
        report.build(a.est, "Phase 3.5 E1: pick for speed, not just safety",
                     "How much lap time did picking the most reliable checkpoint throw away?", a.out,
                     notes=a.notes, sweep=False, extra_sections=[scatter(a.runs)])
        print(a.out)


if __name__ == "__main__":
    main()
