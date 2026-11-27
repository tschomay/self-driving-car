"""Phase 4.7, after E5: how each driver takes sharp corners (owner's E5 explanation, checked against the traces).

    python -m sdc.phase47.corners   -> runs/p47-corners.json  (and a printed summary)

From where.py's 0.1 s traces (5 noisy passes x 150 selection runs). For every sharp turn (|curvature| > 0.15 /m) each
driver reaches: its sharpness (peak curvature), speed 8 m before it and at its start, the slowest speed in it, where
braking starts (first command below -0.2 within 15 m before to 3 m after the turn's start), the largest steering
and the widest offset from the centre line in it, and whether the run ended there. Also, for all driving away from
jumps: how much of the braking happens with no sharp turn in the next 15 m.
"""
from __future__ import annotations

import collections
import json
import pickle
from pathlib import Path

import numpy as np

from sdc.jumps import loop as lp
from sdc.phase47.where import FIELDS

SRC = ("runs/p47-where.pkl", "runs/p47-e4-where.pkl", "runs/p47-e5-where.pkl")
OUT = Path("runs/p47-corners.json")
WHO = ("teacher", "base", "e4", "racer", "e5")
K, BRAKE = 0.15, -0.2
I = {f: i for i, f in enumerate(FIELDS)}


def turns(c):
    a = np.abs(c.kappa) > K
    out, i, n = [], 0, len(a)
    while i < n:
        if a[i] and not a[i - 1]:
            j = i
            while a[j % n]:
                j += 1
            seg = np.abs(c.kappa[np.arange(i, j) % n])
            out.append(dict(s0=i * c.ds, length=(j - i) * c.ds, peak=float(seg.max())))
            i = j
        else:
            i += 1
    return out


def main():
    runs = [r for f in SRC for r in pickle.loads(Path(f).read_bytes()) if r["who"] in WHO]
    cache = {}
    rows = collections.defaultdict(list)
    brake = collections.defaultdict(lambda: [0, 0, 0])     # braking samples: total, no turn within 15 m, within 4 m
    for r in runs:
        c = cache.setdefault(r["course"], lp.make(r["course"]))
        tr = r["trace"]
        if not len(tr):
            continue
        x, v, y, st, th = (tr[:, I[k]] for k in ("x", "v", "y", "steer", "throttle"))
        failed = not r["ep"]["finished"]
        end_x = r["ep"]["x"]
        for t in turns(c):
            for s0 in (t["s0"], t["s0"] + c.length):
                if not ((x >= s0 - 8) & (x <= s0)).any():
                    continue
                if x.max() < s0:                                   # never reached it
                    break
                app = np.abs(x - (s0 - 8)).argmin()
                ent = np.abs(x - s0).argmin()
                inn = (x >= s0) & (x <= s0 + t["length"])
                win = np.flatnonzero((x >= s0 - 15) & (x <= s0 + 3) & (th < BRAKE))
                rows[r["who"]].append(dict(
                    peak=t["peak"], length=t["length"], v_app=float(v[app]), v_entry=float(v[ent]),
                    v_min=float(v[inn].min()) if inn.any() else None,
                    brake_at=float(x[win[0]] - s0) if len(win) else None,
                    steer_max=float(np.abs(st[inn]).max()) if inn.any() else None,
                    y_max=float(np.abs(y[inn]).max()) if inn.any() else None,
                    crashed=bool(failed and s0 - 2 <= end_x <= s0 + t["length"] + 6)))
                break
        jumps = [(j.start - 12, j.end + 3) for j in c.jumps]
        far = np.array([not any(a <= xi % c.length <= b for a, b in jumps) for xi in x]) & (tr[:, I["air"]] == 0)
        for i in np.flatnonzero(far & (th < BRAKE)):
            ahead = np.abs([c.curvature((x[i] + d) % c.length) for d in np.arange(0, 15.01, 0.5)])
            brake[r["who"]][0] += 1
            brake[r["who"]][1] += int(ahead.max() < K)
            brake[r["who"]][2] += int(ahead[:9].max() >= K)
    out = {}
    for who in WHO:
        R = rows[who]
        sharp = [q for q in R if q["peak"] > 0.3]
        mild = [q for q in R if q["peak"] <= 0.3]
        med = lambda xs: round(float(np.median([a for a in xs if a is not None])), 2) if xs else None  # noqa: E731
        pk, ve = np.array([q["peak"] for q in R]), np.array([q["v_entry"] for q in R])
        out[who] = dict(
            turns=len(R),
            v_app=med([q["v_app"] for q in R]), v_entry=med([q["v_entry"] for q in R]),
            v_min=med([q["v_min"] for q in R]),
            brake_at_m=med([q["brake_at"] for q in R]),
            no_brake=round(float(np.mean([q["brake_at"] is None for q in R])), 3),
            entry_vs_sharpness=round(float(np.polyfit(pk, ve, 1)[0]), 2),      # m/s per (1/m) of peak curvature
            entry_mild=med([q["v_entry"] for q in mild]), entry_sharp=med([q["v_entry"] for q in sharp]),
            crash_mild=round(float(np.mean([q["crashed"] for q in mild])), 3),
            crash_sharp=round(float(np.mean([q["crashed"] for q in sharp])), 3),
            steer_max=med([q["steer_max"] for q in R]), y_max=med([q["y_max"] for q in R]),
            braking_samples=brake[who][0],
            braking_no_turn_15m=round(brake[who][1] / max(brake[who][0], 1), 3),
            braking_turn_within_4m=round(brake[who][2] / max(brake[who][0], 1), 3))
        print(who, out[who], flush=True)
    OUT.write_text(json.dumps(dict(summary=out, rows=rows)))


if __name__ == "__main__":
    main()
