"""Phase 4.5 E2 report: python -m sdc.phase45.e2_report --notes "<html>" -> reports/phase45-e2.html

Reads runs/p45-e2/e2.json (python -m sdc.phase45.e2) and re-drives one validation run to draw pitch over time.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from sdc.phase4 import report as rp
from sdc.phase45 import e2
from sdc.phase45 import my_filter as mf

OUT = Path("reports/phase45-e2.html")
SHOW = ("gyro only", "gyro, bias fixed", "accelerometer only", "your filter", "... always blend", "... tolerance ±0.05 g")


def chart(t, series, lo=-40, hi=40):
    """Pitch (deg) vs time (s); series: [(name, y, color index, width)]. Out-of-range values are clipped."""
    W, H, L, R, T, B = 440, 250, 40, 10, 12, 34
    tx = lambda x: L + (W - L - R) * x / t[-1]
    ty = lambda y: T + (H - T - B) * (hi - np.clip(y, lo, hi)) / (hi - lo)
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="pitch over time">']
    for y in range(lo, hi + 1, 20):
        p.append(f'<line x1="{L}" x2="{W - R}" y1="{ty(y):.1f}" y2="{ty(y):.1f}" style="stroke: var(--rule)"/>'
                 f'<text x="{L - 6}" y="{ty(y) + 4:.1f}" text-anchor="end" class="tick">{y}°</text>')
    for x in range(0, int(t[-1]) + 1, 2):
        p.append(f'<text x="{tx(x):.1f}" y="{H - 16}" text-anchor="middle" class="tick">{x}</text>')
    p.append(f'<text x="{(L + W) / 2}" y="{H - 2}" text-anchor="middle" class="tick">time (s)</text>')
    for name, y, c, w in series:
        pts = " ".join(f"{tx(a):.1f},{ty(b):.1f}" for a, b in zip(t, y))
        p.append(f'<polyline points="{pts}" fill="none" style="stroke: var(--s{c})" stroke-width="{w}" '
                 f'stroke-linejoin="round"><title>{html.escape(name)}</title></polyline>')
    p.append("</svg>")
    leg = "".join(f'<div class="run"><span class="sw" style="background: var(--s{c})"></span><span class="name">'
                  f"{html.escape(n)}</span></div>" for n, _, c, _ in series)
    return f'<div class="chart">{"".join(p)}</div><div class="runs">{leg}</div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/p45-e2/e2.json")
    ap.add_argument("--notes", default="")
    ap.add_argument("--course", type=int, default=10003)
    a = ap.parse_args()
    S = json.loads(Path(a.data).read_text())["summary"]
    rows = []
    for name in SHOW:
        r = [name]
        for d in ("nominal", "gate"):
            s = S[d][name]
            r += [f"{s['ground'][0]:.1f}", f"{s['air'][0]:.1f}", f"{s['touchdown_pitch']:.1f}", f"{s['ground'][1]:.1f}",
                  f"{s['air'][1]:.1f}"]
        rows.append(r)
    head = ["estimator"] + [f"{d}: {k}" for d in ("nominal", "gate") for k in
                            ("pitch ground", "pitch air", "pitch touchdown", "roll ground", "roll air")]
    run = e2.log_run(a.course, 0.0, "nominal")
    R = run["rows"]
    t = np.arange(1, len(R) + 1) * 0.02
    f = mf.AttitudeFilter()
    f.reset(run["still_gyro"], run["still_acc"])
    fil = np.degrees([f.update(r[:3], r[3:6])[0] for r in R])
    gb = e2.GyroOnly(True)
    gb.reset(run["still_gyro"], run["still_acc"])
    gyr = np.degrees([gb.update(r[:3], r[3:6])[0] for r in R])
    acc = np.degrees([mf.accel_angles(r[:3])[0] for r in R])
    series = [("accelerometer only", acc, 2, 1), ("gyro, bias fixed", gyr, 3, 1.5), ("your filter", fil, 1, 2),
              ("true pitch", np.degrees(R[:, 6]), 6, 2.5)]
    sections = [f"<h2>Result</h2><div class='note'>{a.notes}</div>",
                "<h2>Errors (degrees, RMS)</h2>" + rp.table(head, rows)
                + "<p class='caption'>The 4.5 expert drives 20 validation courses × 3 starts per physics range (all "
                  "physics, noisy IMU). Each estimator runs on the same IMU logs. Touchdown: the mean absolute pitch "
                  "error at the first step back on the ground. Training range: see e2.json.</p>",
                f"<h2>One run: pitch over time</h2>{chart(t, series)}"
                f"<p class='caption'>Validation course {a.course}, nominal physics, the expert driving; one run, "
                "0.02 s steps. Accelerometer values beyond ±40° are clipped to the edge.</p>"]
    rp.page("Phase 4.5 E2 attitude filter", "Phase 4.5 · E2 · validation courses 10000–10019",
            "How well does your filter know the car's tilt?",
            "Your complementary filter vs. gyro only and accelerometer only, on the realistic IMU.", sections, OUT,
            footer="Code: sdc/phase45/my_filter.py (yours), e2.py, e2_report.py")
    print(OUT)


if __name__ == "__main__":
    main()
