"""Phase 4 report pages: shared page shell, learning-curve chart, GIF embedding, jump tables.

Each experiment has a small script (e1_report.py, ...) that assembles sections and calls page().
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

from sdc.report import CSS, FONTS, line_chart

EXTRA_CSS = """
.note { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: .75rem 1rem; display: grid; gap: .5rem; }
.big2 { font-family: var(--mono); font-size: 1.6rem; }
td.ok { color: var(--s3); } td.bad { color: var(--threshold); }
"""


def gif_tag(path: Path, alt: str) -> str:
    uri = "data:image/gif;base64," + base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img alt="{html.escape(alt)}" src="{uri}">'


def figure(path: Path, caption: str) -> str:
    return f"<figure>{gif_tag(path, caption)}<figcaption>{caption}</figcaption></figure>"


def table(header: list[str], rows: list[list]) -> str:
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table><tr>{th}</tr>{trs}</table></div>'


def curve(runs: list[dict], key="rate", y_label="validation jumps landed (%)", threshold=80.0,
          threshold_label="gate ≥ {:g}%", refs=None) -> str:
    """Validation landing rate vs. training steps, one line per run (metrics.json dicts)."""
    series = []
    for i, m in enumerate(runs):
        ev = m["evals"]
        series.append(dict(x=[e["timesteps"] for e in ev], y=[100 * e[key] for e in ev], color=m.get("_color", i),
                           width=2, dot=True))
    svg = line_chart(series, "training steps", y_label, y_max=100, threshold=threshold,
                     threshold_label=threshold_label)
    return f'<div class="chart">{svg}</div>'


def legend(items: list[tuple[int, str, str]]) -> str:
    """items: (color index, name, big number text)."""
    rows = "".join(f'<div class="run"><span class="sw" style="background: var(--s{c % 6 + 1})"></span>'
                   f'<span class="name">{html.escape(n)}</span><span class="big">{big}</span></div>'
                   for c, n, big in items)
    return f'<div class="runs">{rows}</div>'


def page(title: str, eyebrow: str, h1: str, question: str, sections: list[str], out: str | Path,
         footer: str = "") -> Path:
    body = "".join(f"<section>{s}</section>" for s in sections)
    doc = f"""<title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
{FONTS}
<style>{CSS}{EXTRA_CSS}</style>
<main>
<header><div class="eyebrow">{html.escape(eyebrow)}</div><h1>{html.escape(h1)}</h1>
<p class="question">{question}</p></header>
{body}
<footer>{footer}</footer>
</main>"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    return out
