"""Phone-readable report page for Phase 1 racer runs.

Usage:
    python -m sdc.phase1.report --title "E1 ..." --question "..." --out reports/p1-e1.html \
        runs/p1-pursuit runs/p1-e1-s0 runs/p1-e1-s1 ...

Runs with the same label are one group (seeds), drawn in one color. Headline per
group: lap completion on the 20 held-out tracks, and mean lap time relative to
the pure-pursuit baseline: the two numbers the Phase 1 gate checks.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from sdc.racer import track as trk
from sdc.report import CSS, FONTS, N_SERIES, _fmt_steps, _gif_uri, _smooth, _thin, line_chart

EXTRA_CSS = """
.cell-ok { color: var(--s3); } .cell-bad { color: var(--threshold); }
"""


def _sw(i):
    return f'<span class="sw" style="display:inline-block;background: var(--s{i % N_SERIES + 1})"></span>'


def build(run_dirs, title: str, question: str, out, eyebrow: str = "Phase 1", notes: list[str] | None = None,
          max_gifs: int = 6, curve: str = "laps", gifs_per_group: int = 2, gif_caption: str | None = None,
          footer: str = "sdc/phase1/report.py", extra_sections: list[str] | None = None,
          results_caption: str | None = None, base_name: str = "pure pursuit", extra_sub=None,
          eval_threshold: float = 95, eval_caption: str | None = None, extra_eval=None) -> Path:
    runs = []
    for d in run_dirs:
        m = json.loads((Path(d) / "metrics.json").read_text())
        m["_dir"] = Path(d)
        runs.append(m)
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for m in runs:
        groups.setdefault(m["label"], []).append(m)
    color = {label: i for i, label in enumerate(groups)}
    base = runs[0]["baseline"]
    base_time = base["mean_lap_time"]

    rows = []
    for label, ms in groups.items():
        comp = [m["final_eval"]["completion"] for m in ms]
        ratios = [m["compare"]["lap_time_ratio"] for m in ms]
        gates = sum(m["compare"]["gate"] for m in ms)
        sub = []
        times = [m["final_eval"]["mean_lap_time"] for m in ms if m["final_eval"]["mean_lap_time"]]
        if times:
            r_ok = [r for r in ratios if r]
            sub.append(f"mean lap {np.mean(times):.1f} s on finished tracks · {np.mean(r_ok):.2f}× {base_name}")
        else:
            sub.append("no held-out lap finished")
        if len(ms) > 1:
            sub.append("per seed: " + ", ".join(f"{100 * c:.0f}%" for c in comp))
        if "train_tracks_eval" in ms[0]:
            tc = [m["train_tracks_eval"]["completion"] for m in ms]
            sub.append(f"on its own training track(s): {100 * np.mean(tc):.0f}% completion")
        if "train_wall_s" in ms[0]:
            sub.append(f"train: {np.mean([m['trained_timesteps'] for m in ms]):,.0f} steps, "
                       f"{np.mean([m['train_wall_s'] for m in ms]):.0f} s")
        if extra_sub is not None:
            sub.extend(extra_sub(ms))
        if ms[0]["preset"] == "pursuit":
            pill = '<span class="pill">baseline</span>'
        else:
            pill = f'<span class="pill {"pass" if gates else "fail"}">gate {gates}/{len(ms)}</span>'
        rows.append(f'<div class="run"><span class="sw" style="background: var(--s{color[label] % N_SERIES + 1})"></span>'
                    f'<span class="name">{html.escape(label)}{pill}</span>'
                    f'<span class="big">{100 * np.mean(comp):.0f}%</span>'
                    f'<span class="sub">{"<br>".join(html.escape(s) for s in sub)}</span></div>')

    cap = results_caption or (f'Big number: share of the 20 held-out tracks '
                f'(never trained on) where the car finished one lap from a standing start without leaving the '
                f'road. Gate: ≥ 95% and mean lap time ≤ 1.1× the hand-coded pure-pursuit driver '
                f'({base_time:.1f} s).')
    sections = [f'<section><h2>Results</h2><p class="caption">{html.escape(cap)}</p><div class="runs">{"".join(rows)}</div></section>']

    key = {"laps": ("laps per training episode", "laps"), "return": ("episode return", "ret"),
           "speed": ("mean speed (m/s)", "speed")}[curve]
    train_series, eval_series = [], []
    y_hi = 0.0
    for label, ms in groups.items():
        for m in ms:
            if not m["episodes"]:
                continue
            x = [e["timesteps"] for e in m["episodes"]]
            y = [e[key[1]] for e in m["episodes"]]
            sx_, sy_ = _thin(x, list(_smooth(y, 50)))
            y_hi = max(y_hi, max(sy_))
            train_series.append(dict(x=sx_, y=sy_, color=color[label], width=2 if len(ms) == 1 else 1.5,
                                     opacity=1 if len(ms) == 1 else .75, dot=True))
            if m["evals"]:
                eval_series.append(dict(x=[e["timesteps"] for e in m["evals"]],
                                        y=[100 * e["completion"] for e in m["evals"]], color=color[label],
                                        width=2 if len(ms) == 1 else 1.5, opacity=1 if len(ms) == 1 else .75, dot=True))
                if extra_eval is not None:
                    eval_series.extend(extra_eval(m, color[label]))
    if train_series:
        y_max = max(1.0, float(np.ceil(y_hi * 1.15 * 2) / 2)) if curve == "laps" else y_hi * 1.15
        sections.append('<section><h2>Learning curve</h2>'
                        f'<div class="chart">{line_chart(train_series, "environment steps", key[0], y_max=y_max, threshold=None)}</div>'
                        f'<p class="caption">{html.escape(key[0].capitalize())}, 50-episode moving average. Training '
                        'episodes start at a random point and end on leaving the road or after 100 s, using the '
                        'exploring (random-noise) policy.</p></section>')
    if eval_series:
        gaps = [m["evals"][2]["timesteps"] - m["evals"][1]["timesteps"] for m in runs if len(m.get("evals", [])) > 2]
        sections.append('<section><h2>Held-out lap completion during training</h2>'
                        f'<div class="chart">{line_chart(eval_series, "environment steps", "held-out completion (%)", y_max=105, threshold=eval_threshold, threshold_label="gate ≥ {:g}%")}</div>'
                        f'<p class="caption">Every {_fmt_steps(float(np.median(gaps)) if gaps else 0)} steps: one lap '
                        'attempt on each of the 20 held-out tracks with the greedy (no-noise) policy.'
                        f'{(" " + html.escape(eval_caption)) if eval_caption else ""}</p></section>')

    figs = []
    for label, ms in groups.items():
        m = ms[0]
        for g in m.get("gifs", [])[:gifs_per_group]:
            uri = _gif_uri(m["_dir"] / g["file"])
            if not uri or len(figs) >= max_gifs:
                continue
            res = (f"finished in {g['lap_time']:.1f} s" if g["finished"]
                   else f"{'left the road' if g['crashed'] else 'timed out'} at {100 * g['laps']:.0f}% of the lap")
            who = f"{label}{f', seed {m['seed']}' if len(ms) > 1 else ''}"
            figs.append(f'<figure><img src="{uri}" alt="Lap by {html.escape(who)} on track {g["track"]}">'
                        f'<figcaption>{_sw(color[label])} {html.escape(who)} · '
                        f'{"held-out" if g["track"] >= trk.HELD_OUT_BASE else "its own training"} track {g["track"]}: '
                        f'{res}</figcaption></figure>')
    if figs:
        cap = gif_caption or ('Each GIF is one lap attempt on one held-out '
                        'track, from a standing start, at real speed (10 steps per second). Blue lines are the 11 '
                        'sensor rays, the only thing the RL policy sees besides its speed. The trail shows speed '
                        '(blue slow, orange fast). The bar along the bottom fills with lap distance. An end card '
                        'says how it ended, then the GIF loops. Where a row has several seeds, GIFs show the first '
                        'seed; a second GIF, if present, is the first held-out track it failed.')
        sections.append(f'<section><h2>Laps</h2><p class="caption">{html.escape(cap)}</p>'
                        f'<div class="gifs">{"".join(figs)}</div></section>')

    # per-track table: baseline vs each group's first seed... use all seeds' completion count
    head = "".join(f"<th>{_sw(color[label])} {html.escape(label)}</th>" for label in groups)
    trs = []
    for k, brow in enumerate(base["tracks"]):
        cells = []
        for label, ms in groups.items():
            rs = [m["final_eval"]["tracks"][k] for m in ms]
            fin = [r for r in rs if r["finished"]]
            if len(ms) == 1:
                r = rs[0]
                txt = f"{r['lap_time']:.1f} s" if r["finished"] else ("off road" if r["crashed"] else "timeout") + f" {100 * r['laps']:.0f}%"
            else:
                txt = f"{len(fin)}/{len(ms)}" + (f" · {np.mean([r['lap_time'] for r in fin]):.1f} s" if fin else "")
            cells.append(f'<td class="{"cell-ok" if len(fin) == len(ms) else "cell-bad"}">{txt}</td>')
        trs.append(f"<tr><td>{brow['track']}</td>{''.join(cells)}</tr>")
    sections.append('<section><h2>Per held-out track</h2><p class="caption">Lap time, or how far it got before '
                    'leaving the road. With several seeds: seeds that finished, and their mean lap time.</p>'
                    f'<div class="tablewrap"><table><tr><th>track</th>{head}</tr>{"".join(trs)}</table></div></section>')

    sections.extend(extra_sections or [])
    if notes:
        sections.append('<section class="notes"><h2>Notes</h2>' + "".join(f"<p>{html.escape(n)}</p>" for n in notes)
                        + "</section>")

    cfg_rows = []
    for label, ms in groups.items():
        m = ms[0]
        if m["preset"] == "pursuit":
            continue
        ppo = ", ".join(f"{k}={v}" for k, v in m.get("ppo", {}).items()) or "SB3 defaults"
        env = ", ".join(f"{k}={v}" for k, v in m.get("env", {}).items())
        cfg_rows.append(f"<tr><td>{html.escape(label)}</td><td>{m.get('n_envs', '–')}</td>"
                        f"<td>{', '.join(str(x['seed']) for x in ms)}</td><td>{html.escape(ppo)}</td>"
                        f"<td>{html.escape(env)}</td></tr>")
    if cfg_rows:
        sections.append('<section><h2>Configs</h2><div class="tablewrap"><table><tr><th>run</th><th>cars</th>'
                        f'<th>seeds</th><th>PPO settings</th><th>racer settings</th></tr>{"".join(cfg_rows)}</table></div></section>')

    page = (f'<title>{html.escape(title)}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            f'{FONTS}\n<style>{CSS}{EXTRA_CSS}</style>\n<main><header><span class="eyebrow">{html.escape(eyebrow)} · 2D racer</span>'
            f'<h1>{html.escape(title)}</h1><p class="question">{html.escape(question)}</p></header>'
            f'{"".join(sections)}<footer>Generated by {footer} · CPU only</footer></main>')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--title", required=True)
    p.add_argument("--question", default="")
    p.add_argument("--eyebrow", default="Phase 1")
    p.add_argument("--note", action="append", default=[])
    p.add_argument("--curve", default="laps", choices=["laps", "return", "speed"])
    p.add_argument("--gifs-per-group", type=int, default=2)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(build(a.runs, a.title, a.question, a.out, a.eyebrow, a.note, curve=a.curve, gifs_per_group=a.gifs_per_group))


if __name__ == "__main__":
    main()
