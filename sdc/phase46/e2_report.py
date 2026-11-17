"""Phase 4.6 E2 report: python -m sdc.phase46.e2_report --notes "<html>" -> reports/phase46-e2.html

Reads runs/p46-e2/e2.json and paths-10000.pkl (python -m sdc.phase46.e2), and runs/p46-e2-walk01/e2.json (the same
with a tenth of the gyro's bias drift) if it exists.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sdc.gifs import _font
from sdc.phase4 import report as rp

OUT = Path("reports/phase46-e2.html")
OWNER = "owner's (gyro, tilt-corrected, hold in the air)"
TRUE, RAW, CLOSED = (120, 130, 142), (200, 70, 50), (31, 111, 235)


def med(rows, f):
    return float(np.median([f(r) for r in rows]))


def path_figure(keep, kind, title):
    k = keep[kind]
    t, e, c = k["truth"][:, 2:4], k["est"][:, :2], k["closed"][:, 1:3]
    pts = np.vstack([t, e, c])
    lo, hi = pts.min(0) - 2, pts.max(0) + 2
    W = 400
    sc = W / (hi - lo).max()
    H = int((hi - lo)[1] * sc) + 46
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    px = lambda p: [((x - lo[0]) * sc, H - 6 - (y - lo[1]) * sc) for x, y in p]  # noqa: E731
    for p, col, w in ((t, TRUE, 5), (e, RAW, 2), (c, CLOSED, 2)):
        d.line(px(p[::3]), fill=col, width=w)
    x0, y0 = px(t[:1])[0]
    d.ellipse([x0 - 5, y0 - 5, x0 + 5, y0 + 5], fill=(23, 32, 43))
    d.text((8, 4), title, font=_font(13, bold=True), fill=(23, 32, 43))
    for i, (name, col) in enumerate((("true path", TRUE), ("estimate", RAW), ("after the finish-line correction", CLOSED))):
        d.rectangle([8 + i * 0, 24 + i * 0, 8, 24], fill=col)
    xx = 8
    for name, col in (("true path", TRUE), ("estimate", RAW), ("closed at the line", CLOSED)):
        d.rectangle([xx, 26, xx + 12, 34], fill=col)
        d.text((xx + 16, 30), name, font=_font(11), fill=(91, 104, 120), anchor="lm")
        xx += 28 + int(d.textlength(name, font=_font(11)))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    D = json.loads(Path("runs/p46-e2/e2.json").read_text())
    keep = pickle.load(open("runs/p46-e2/paths-10000.pkl", "rb"))
    lap = {"slow": "slow lap (2 m/s)", "race": "race lap (teacher)"}
    main_rows, var_rows, map_rows = [], [], []
    for kind in ("slow", "race"):
        R = D[kind]["variants"][OWNER]
        main_rows.append([lap[kind], f"{D[kind]['lap_s']:.0f} s", f"{D[kind]['finished']}/{D[kind]['runs']}",
                          f"<b>{med(R, lambda r: r['closing']):.1f} m</b>",
                          f"<b>{med(R, lambda r: 100 * r['closing'] / r['driven']):.1f}%</b>",
                          f"{med(R, lambda r: abs(r['heading_deg'])):.0f}°", f"{med(R, lambda r: r['distance_pct']):+.1f}%",
                          f"{med(R, lambda r: r['heading_only']):.1f} m", f"{med(R, lambda r: r['distance_only']):.1f} m"])
        for name, rows in D[kind]["variants"].items():
            var_rows.append([lap[kind], name, f"{med(rows, lambda r: r['closing']):.1f} m",
                             f"{med(rows, lambda r: abs(r['heading_deg'])):.0f}°",
                             f"{med(rows, lambda r: r['distance_pct']):+.1f}%"])
        T = [t for r in R for t in r["turns"]]
        w = np.array([t["where"] for t in T])
        ratio = np.array([t["radius_map"] / t["radius_true"] for t in T])
        dv = np.abs(np.sqrt(3.5 * np.array([t["radius_map"] for t in T])) - np.sqrt(3.5 * np.array([t["radius_true"] for t in T])))
        map_rows.append([lap[kind], len(T), f"{med(R, lambda r: r['mean_err_raw']):.1f} m → {med(R, lambda r: r['mean_err_closed']):.1f} m",
                         f"{np.median(np.abs(w)):.1f} m (9 in 10 within {np.percentile(np.abs(w), 90):.1f} m)",
                         f"{np.median(w):+.1f} m", f"×{np.median(ratio):.2f}", f"{np.median(dv):.2f} m/s"])
    sections = [f"<h2>Result</h2><div class='note'>{a.notes}</div>",
                "<h2>Drift over one lap (your estimator)</h2>" + rp.table(
                    ["lap", "lap time", "laps finished", "position off at the line", "as share of distance driven",
                     "heading off", "distance off", "position off from heading alone", "from distance alone"], main_rows)
                + "<p class='caption'>Medians over the finished laps: 20 validation loops × 3 starts, gate-range physics, "
                  "parts-list sensors. Before the finish-line correction. \"From heading alone\": the true distance with "
                  "your estimated heading; \"from distance alone\": the true heading with your estimated distance.</p>",
                "<h2>One lap of loop 10000</h2><div class='gifs'>"
                + "".join(f'<figure><img alt="{lap[k]} paths" src="{path_figure(keep, k, lap[k])}"><figcaption>{cap}</figcaption></figure>'
                          for k, cap in (("slow", "Slow lap, start 0, gate-range physics. The dot is the start line."),
                                         ("race", "Race lap, same loop and physics.")))
                + "</div>",
                "<h2>Your choices against the alternatives</h2>" + rp.table(
                    ["lap", "estimator", "position off at the line", "heading off", "distance off"], var_rows)
                + "<p class='caption'>The same sensor logs replayed with one choice changed at a time.</p>",
                "<h2>The map of the turns (after your finish-line correction)</h2>" + rp.table(
                    ["lap", "turns", "mean position error: before → after closing", "a mapped turn is off by",
                     "typical shift", "mapped radius / true", "corner speed error"], map_rows)
                + "<p class='caption'>Each true turn's tightest point against the tightest point of the mapped curvature "
                  "within 10 m. Positive shift: the map puts the turn later than it is. Corner speed: the speed for "
                  "3.5 m/s² sideways at the mapped radius vs the true one.</p>"]
    w01 = Path("runs/p46-e2-walk01/e2.json")
    if w01.exists():
        E = json.loads(w01.read_text())
        rows = []
        for kind in ("slow", "race"):
            for label, X in (("as modelled since Phase 4.5 (0.05°/s per √s)", D), ("a tenth of that", E)):
                R = X[kind]["variants"][OWNER]
                T = [t for r in R for t in r["turns"]]
                rows.append([lap[kind], label, f"{med(R, lambda r: r['closing']):.1f} m",
                             f"{med(R, lambda r: abs(r['heading_deg'])):.0f}°",
                             f"{np.median(np.abs([t['where'] for t in T])):.1f} m"])
        sections.append("<h2>Check: how much of this is the gyro's modelled bias drift?</h2>" + rp.table(
            ["lap", "gyro bias drift", "position off at the line", "heading off", "a mapped turn is off by"], rows)
            + "<p class='caption'>The sim's gyro bias wanders by 0.05°/s per √second (× the noise scale, 0.5–2 at gate "
              "range), set in Phase 4.5 when runs lasted 12 s. Over an 87 s lap that adds up to tens of degrees. "
              "Same laps and seeds, only the drift changed.</p>")
    rp.page("Phase 4.6 E2 dead reckoning", "Phase 4.6 · E2 · validation loops 10000–10019", "Your dead reckoning",
            "How far does the position estimate drift over one lap, and is the map of the turns it draws usable?",
            sections, OUT, footer="Code: sdc/phase46/my_deadreckoning.py, e2.py, e2_report.py, laps.py")
    print(OUT)


if __name__ == "__main__":
    main()
