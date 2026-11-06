"""Build a phone-readable HTML report page from one or more run directories.

Usage:
    python -m sdc.report --title "E1 Random vs PPO" --question "..." \
        --out reports/e1.html runs/e1-random runs/e1-ppo-s0 ...

Runs with the same `label` in metrics.json are treated as one group (e.g. seeds)
and drawn in the same color. The page is self-contained: inline SVG charts and the
rollout GIFs embedded as data URIs, so it can be published as a single artifact.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

N_SERIES = 6

CSS = """
/* Layout: one narrow lab-notebook column; numbers first, then curves, then the car itself. */
:root {
  --bg: #f5f7f9; --surface: #ffffff; --fg: #17202b; --muted: #5b6878; --rule: #d9e0e8;
  --accent: #1f6feb; --threshold: #b4462b;
  --s1: #1f6feb; --s2: #c2410c; --s3: #0f8a6a; --s4: #8b3fd1; --s5: #a16207; --s6: #5b6878;
  --display: "Bricolage Grotesque", "Avenir Next", system-ui, sans-serif;
  --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f141a; --surface: #161d26; --fg: #e4e9ef; --muted: #93a1b2; --rule: #2a3542;
  --accent: #6aa5ff; --threshold: #f08a6c;
  --s1: #6aa5ff; --s2: #f59a5b; --s3: #4fd1a5; --s4: #c08cff; --s5: #e3b341; --s6: #93a1b2;
  color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0f141a; --surface: #161d26; --fg: #e4e9ef; --muted: #93a1b2; --rule: #2a3542;
  --accent: #6aa5ff; --threshold: #f08a6c;
  --s1: #6aa5ff; --s2: #f59a5b; --s3: #4fd1a5; --s4: #c08cff; --s5: #e3b341; --s6: #93a1b2;
  color-scheme: dark; }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 16px; line-height: 1.5; }
main { max-width: 44rem; margin: 0 auto; display: grid; gap: 2rem;
  padding-block: 1.5rem 3rem;
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); }
header { display: grid; gap: .5rem; }
.eyebrow { font-family: var(--mono); font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
h1 { font-family: var(--display); font-weight: 700; font-size: clamp(1.6rem, 6vw, 2.2rem); line-height: 1.1; margin: 0; text-wrap: balance; }
h2 { font-family: var(--display); font-weight: 600; font-size: 1.15rem; margin: 0; text-wrap: balance; }
p { margin: 0; max-width: 65ch; }
.question { color: var(--muted); }
section { display: grid; gap: .75rem; min-width: 0; }
.caption { color: var(--muted); font-size: .875rem; }
.runs { display: grid; gap: .5rem; }
.run { display: grid; grid-template-columns: .75rem 1fr auto; gap: .25rem .75rem; align-items: baseline;
  padding-block: .6rem; border-bottom: 1px solid var(--rule); }
.run:last-child { border-bottom: 0; }
.sw { width: .75rem; height: .75rem; border-radius: 2px; align-self: center; }
.run .name { font-weight: 600; min-width: 0; }
.run .big { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 1.25rem; text-align: right; }
.run .sub { grid-column: 2 / 4; color: var(--muted); font-size: .85rem; font-family: var(--mono); font-variant-numeric: tabular-nums; }
.pill { font-family: var(--mono); font-size: .7rem; padding: .05rem .4rem; border-radius: 99px; border: 1px solid currentColor; margin-left: .4rem; white-space: nowrap; }
.pass { color: var(--s3); } .fail { color: var(--threshold); }
.chart { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: .5rem; overflow-x: auto; }
.chart svg { display: block; width: 100%; height: auto; min-width: 320px; }
.ax { stroke: var(--rule); stroke-width: 1; }
.tick { fill: var(--muted); font: 11px var(--mono); }
.axlabel { fill: var(--muted); font: 12px var(--body); }
.thr { stroke: var(--threshold); stroke-dasharray: 4 4; stroke-width: 1.25; fill: none; }
.thrlabel { fill: var(--threshold); font: 11px var(--mono); }
.gifs { display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); gap: 1rem; }
figure { margin: 0; display: grid; gap: .35rem; min-width: 0; }
figure img { width: 100%; border: 1px solid var(--rule); border-radius: 6px; background: #fff; }
figcaption { font-size: .85rem; color: var(--muted); }
.notes { display: grid; gap: .5rem; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; font-family: var(--mono); font-size: .8rem; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: .3rem .75rem .3rem 0; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th { color: var(--muted); font-weight: 500; }
footer { color: var(--muted); font-size: .8rem; font-family: var(--mono); }
"""

FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700'
         '&family=JetBrains+Mono:wght@400;500&family=Source+Sans+3:wght@400;600&display=swap">')


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = max(hi - lo, 1e-9)
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = np.ceil(lo / step) * step
    return [float(t) for t in np.arange(start, hi + step * 1e-6, step)]


def _fmt_steps(x: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:g}M"
    if x >= 1e3:
        return f"{x / 1e3:g}k"
    return f"{x:g}"


def line_chart(series: list[dict], x_label: str, y_label: str, y_max: float = 520, threshold: float | None = 475,
               floor: float | None = None) -> str:
    """series: [{x: [...], y: [...], color: idx, width: float, opacity: float}] drawn on one shared scale."""
    W, H, L, R, T, B = 440, 260, 44, 10, 14, 40
    xs = [x for s in series for x in s["x"]]
    x_hi = max(xs) if xs else 1
    x_ticks = _nice_ticks(0, x_hi)
    x_hi = max(x_hi, x_ticks[-1])
    y_ticks = [t for t in _nice_ticks(0, y_max) if t <= y_max]

    def sx(x):
        return L + (W - L - R) * x / x_hi

    def sy(y):
        return T + (H - T - B) * (1 - min(y, y_max) / y_max)

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(y_label)} vs {html.escape(x_label)}">']
    for t in y_ticks:
        parts.append(f'<line class="ax" x1="{L}" x2="{W - R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke-opacity=".6"/>')
        parts.append(f'<text class="tick" x="{L - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    for t in x_ticks:
        parts.append(f'<text class="tick" x="{sx(t):.1f}" y="{H - B + 16}" text-anchor="middle">{_fmt_steps(t)}</text>')
    parts.append(f'<line class="ax" x1="{L}" x2="{W - R}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>')
    parts.append(f'<text class="axlabel" x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle">{html.escape(x_label)}</text>')
    parts.append(f'<text class="axlabel" x="12" y="{(T + H - B) / 2}" text-anchor="middle" '
                 f'transform="rotate(-90 12 {(T + H - B) / 2})">{html.escape(y_label)}</text>')
    if threshold is not None:
        parts.append(f'<line class="thr" x1="{L}" x2="{W - R}" y1="{sy(threshold):.1f}" y2="{sy(threshold):.1f}"/>')
        parts.append(f'<text class="thrlabel" x="{W - R}" y="{sy(threshold) - 5:.1f}" text-anchor="end">solved ≥ {threshold:g}</text>')
    if floor is not None:
        parts.append(f'<line class="ax" x1="{L}" x2="{W - R}" y1="{sy(floor):.1f}" y2="{sy(floor):.1f}" '
                     f'stroke-dasharray="2 3" style="stroke: var(--s6)"/>')
        parts.append(f'<text class="tick" x="{W - R}" y="{sy(floor) - 5:.1f}" text-anchor="end">random ≈ {floor:.0f}</text>')
    for s in series:
        if len(s["x"]) == 0:
            continue
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(s["x"], s["y"]))
        parts.append(f'<polyline points="{pts}" fill="none" style="stroke: var(--s{s["color"] % N_SERIES + 1})" '
                     f'stroke-width="{s.get("width", 2)}" stroke-opacity="{s.get("opacity", 1)}" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
        if s.get("dot"):
            x, y = s["x"][-1], s["y"][-1]
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" style="fill: var(--s{s["color"] % N_SERIES + 1})"/>')
    parts.append("</svg>")
    return "".join(parts)


def _smooth(y: list[float], k: int = 20) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y
    c = np.cumsum(np.insert(y, 0, 0))
    out = np.empty_like(y)
    for i in range(len(y)):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def _thin(x, y, n=400):
    if len(x) <= n:
        return list(x), list(y)
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return [x[i] for i in idx], [y[i] for i in idx]


def _gif_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    return "data:image/gif;base64," + base64.b64encode(path.read_bytes()).decode()


def build(run_dirs: list[str | Path], title: str, question: str, out: str | Path, eyebrow: str = "Phase 0",
          notes: list[str] | None = None, max_gifs: int = 4) -> Path:
    runs = []
    for d in run_dirs:
        d = Path(d)
        m = json.loads((d / "metrics.json").read_text())
        m["_dir"] = d
        runs.append(m)

    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for m in runs:
        groups.setdefault(m["label"], []).append(m)
    color = {label: i for i, label in enumerate(groups)}

    random_floor = next((g[0]["final_eval"]["mean"] for g in groups.values() if g[0]["preset"] == "random"), None)

    # Summary rows, one per group.
    rows = []
    for label, ms in groups.items():
        finals = [m["final_eval"]["mean"] for m in ms]
        ok = sum(f >= m["solved_threshold"] for f, m in zip(finals, ms))
        solved = [m["solved_at"] for m in ms if m.get("solved_at")]
        sub = [f"held-out: {np.mean(finals):.1f} mean over {ms[0]['final_eval']['episodes']} eps"
               + (f" × {len(ms)} seeds" if len(ms) > 1 else "")]
        if len(ms) > 1:
            sub.append("per seed: " + ", ".join(f"{f:.0f}" for f in finals))
        if "train_wall_s" in ms[0]:
            sub.append(f"train: {np.mean([m['trained_timesteps'] for m in ms]):,.0f} steps, "
                       f"{np.mean([m['train_wall_s'] for m in ms]):.0f} s")
        if solved:
            sub.append(f"first solved eval: {np.median([s['timesteps'] for s in solved]):,.0f} steps / "
                       f"{np.median([s['wall_s'] for s in solved]):.0f} s (median of {len(solved)}/{len(ms)})")
        pill = (f'<span class="pill pass">{ok}/{len(ms)} solved</span>' if ok
                else f'<span class="pill fail">0/{len(ms)} solved</span>')
        rows.append(
            f'<div class="run"><span class="sw" style="background: var(--s{color[label] % N_SERIES + 1})"></span>'
            f'<span class="name">{html.escape(label)}{pill}</span>'
            f'<span class="big">{np.mean(finals):.0f}</span>'
            f'<span class="sub">{"<br>".join(html.escape(s) for s in sub)}</span></div>')

    # Training curve: raw episode returns (faint) + 20-episode moving average.
    train_series, eval_series = [], []
    for label, ms in groups.items():
        for m in ms:
            if not m["episodes"]:
                continue
            x = [e["timesteps"] for e in m["episodes"]]
            y = [e["ret"] for e in m["episodes"]]
            if len(ms) == 1:
                tx, ty = _thin(x, y)
                train_series.append(dict(x=tx, y=ty, color=color[label], width=1, opacity=.18))
            sx_, sy_ = _thin(x, list(_smooth(y)))
            train_series.append(dict(x=sx_, y=sy_, color=color[label], width=2 if len(ms) == 1 else 1.5,
                                     opacity=1 if len(ms) == 1 else .75, dot=True))
            if m["evals"]:
                eval_series.append(dict(x=[e["timesteps"] for e in m["evals"]], y=[e["mean"] for e in m["evals"]],
                                        color=color[label], width=2 if len(ms) == 1 else 1.5,
                                        opacity=1 if len(ms) == 1 else .75, dot=True))

    gaps = [m["evals"][2]["timesteps"] - m["evals"][1]["timesteps"] for m in runs if len(m.get("evals", [])) > 2]
    eval_gap = float(np.median(gaps)) if gaps else 5000.0
    sections = []
    sections.append(f'<section><h2>Results</h2><p class="caption">Big number: mean return on 100 held-out '
                    f'episodes (reset seeds the training never used). CartPole caps an episode at 500; '
                    f'"solved" means a mean of at least 475.</p><div class="runs">{"".join(rows)}</div></section>')
    if train_series:
        sections.append(
            '<section><h2>Learning curve</h2>'
            f'<div class="chart">{line_chart(train_series, "environment steps", "episode return", floor=random_floor)}</div>'
            '<p class="caption">Return of each training episode (faint) and its 20-episode moving average. '
            'Training episodes use the exploring, stochastic policy.</p></section>')
    if eval_series:
        sections.append(
            '<section><h2>Held-out evaluation during training</h2>'
            f'<div class="chart">{line_chart(eval_series, "environment steps", "mean eval return", floor=random_floor)}</div>'
            f'<p class="caption">Every {_fmt_steps(eval_gap)} steps: mean of 10 held-out episodes with the greedy (deterministic) policy.</p>'
            '</section>')

    figs = []
    for label, ms in groups.items():
        for m in ms[:1]:
            uri = _gif_uri(m["_dir"] / "rollout.gif")
            if uri and len(figs) < max_gifs:
                figs.append(f'<figure><img src="{uri}" alt="Rollout of {html.escape(label)}">'
                            f'<figcaption><span class="sw" style="display:inline-block;background: var(--s{color[label] % N_SERIES + 1})"></span> '
                            f'{html.escape(label)}{f", seed {m['seed']}" if len(ms) > 1 else ""}: return {m["gif_return"]:.0f}</figcaption></figure>')
    if figs:
        sections.append('<section><h2>Rollouts</h2><p class="caption">Each GIF is one held-out episode by the '
                        'final policy, played at real speed until it ends. The bar along the bottom fills toward '
                        'the 500-step cap. An end card says why the episode stopped, then the GIF loops. '
                        'Where a row has several seeds, the GIF shows the first one.</p>'
                        f'<div class="gifs">{"".join(figs)}</div></section>')

    if notes:
        sections.append('<section class="notes"><h2>Notes</h2>' + "".join(f"<p>{html.escape(n)}</p>" for n in notes) + "</section>")

    cfg_rows = []
    for label, ms in groups.items():
        m = ms[0]
        ppo = ", ".join(f"{k}={v[1] if isinstance(v, list) else v}{' (linear decay)' if isinstance(v, list) else ''}"
                        for k, v in m.get("ppo", {}).items()) or "SB3 defaults"
        cfg_rows.append(f"<tr><td>{html.escape(label)}</td><td>{m['preset']}</td><td>{m.get('n_envs', '–')}</td>"
                        f"<td>{', '.join(str(x['seed']) for x in ms)}</td><td>{html.escape(ppo)}</td></tr>")
    sections.append('<section><h2>Configs</h2><div class="tablewrap"><table><tr><th>run</th><th>preset</th>'
                    f'<th>envs</th><th>seeds</th><th>PPO overrides</th></tr>{"".join(cfg_rows)}</table></div></section>')

    page = (f'<title>{html.escape(title)}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            f'{FONTS}\n<style>{CSS}</style>\n<main><header><span class="eyebrow">{html.escape(eyebrow)} · CartPole-v1</span>'
            f'<h1>{html.escape(title)}</h1><p class="question">{html.escape(question)}</p></header>'
            f'{"".join(sections)}<footer>Generated by sdc/report.py · PPO from Stable-Baselines3 · CPU only</footer></main>')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--title", required=True)
    p.add_argument("--question", default="")
    p.add_argument("--eyebrow", default="Phase 0")
    p.add_argument("--note", action="append", default=[])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(build(a.runs, a.title, a.question, a.out, a.eyebrow, a.note))


if __name__ == "__main__":
    main()
