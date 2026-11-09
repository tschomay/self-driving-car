"""Phone-readable report page for Phase 2 (camera) runs, built on the Phase 1 report.

Usage:
    python -m sdc.phase2.report --title "E1 ..." --question "..." --out reports/p2-e1.html \
        runs/p1-pursuit runs/p2-e1-s0 runs/p2-e1-s1 ...
"""

from __future__ import annotations

import argparse

from sdc.phase1 import report as p1

GIF_CAPTION = ("Each GIF is one lap attempt on one held-out track, from a standing start, at real speed (10 steps "
               "per second). The inset (bottom right) is what the camera car sees at that moment: the 64×64 image "
               "its network gets, enlarged, with a red triangle where the car sits. The blue rays on the map are "
               "only drawn for comparison; the camera car does not get them. The trail shows speed (blue slow, "
               "orange fast). An end card says how it ended, then the GIF loops. GIFs show the first seed of a "
               "row; a second GIF, if present, is the first held-out track it failed.")


def saliency_section(figures: list[tuple[str, str]], caption: str) -> str:
    """HTML section with saliency figures (path, caption) embedded as data URIs."""
    import base64
    import html

    figs = "".join(f'<figure><img src="data:image/png;base64,{base64.b64encode(open(p, "rb").read()).decode()}" '
                   f'alt="{html.escape(c)}" style="width:100%;height:auto"><figcaption>{html.escape(c)}</figcaption>'
                   '</figure>' for p, c in figures)
    return (f'<section><h2>What it looks at</h2><p class="caption">{html.escape(caption)}</p>'
            f'<div style="display:grid;gap:16px;min-width:0">{figs}</div></section>')


def build(run_dirs, title, question, out, notes=None, saliency=None, saliency_caption="", **kw):
    extra = [saliency_section(saliency, saliency_caption)] if saliency else None
    return p1.build(run_dirs, title, question, out, eyebrow="Phase 2", notes=notes, gif_caption=GIF_CAPTION,
                    footer="sdc/phase2/report.py", extra_sections=extra, **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--title", required=True)
    p.add_argument("--question", default="")
    p.add_argument("--note", action="append", default=[])
    p.add_argument("--curve", default="laps", choices=["laps", "return", "speed"])
    p.add_argument("--gifs-per-group", type=int, default=2)
    p.add_argument("--saliency", action="append", default=[], metavar="PNG=CAPTION")
    p.add_argument("--saliency-caption", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sal = [tuple(x.split("=", 1)) for x in a.saliency]
    print(build(a.runs, a.title, a.question, a.out, a.note, saliency=sal, saliency_caption=a.saliency_caption,
                curve=a.curve, gifs_per_group=a.gifs_per_group))


if __name__ == "__main__":
    main()
