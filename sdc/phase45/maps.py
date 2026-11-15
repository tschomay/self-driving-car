"""E6: out-lap maps for the timed lap, from walk-through crawls (sdc/phase45/outlap.py: pitch shape + ToF scan).

    python -m sdc.phase45.maps --rows runs/p45-walk-rows-scan.pkl

Writes runs/p45-maps-train.json (practice courses 100000-100299, training-range physics) and runs/p45-maps-val.json
(validation courses x 3 starts, gate-range physics): {"course|x0": [[lip odometer m, mapped speed m/s], ...]}.
Practice-course maps are cross-fitted (two halves; each half's speeds come from the estimator fitted on the other),
so the training maps are as noisy as maps of unseen courses. Validation maps use the estimator fitted on all practice
courses. The estimator is ridge regression on the scan features, ridge 10 (E3c/E5 test).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from sdc.phase45 import outlap as ol


def build(rows, model):
    maps = {}
    for r in rows:
        key = f"{r['course']}|{r['x0']:.2f}"
        maps.setdefault(key, []).append([round(r["meas"]["lip"], 3), round(ol.predict_walk(model, r["meas"]), 3)])
    return {k: sorted(v) for k, v in maps.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="runs/p45-walk-rows-scan.pkl")
    ap.add_argument("--ridge", type=float, default=10.0)
    a = ap.parse_args()
    D = pickle.load(open(a.rows, "rb"))
    tr, va = D["tr"], D["va"]
    courses = sorted({r["course"] for r in tr})
    half = set(courses[::2])
    A = [r for r in tr if r["course"] in half]
    B = [r for r in tr if r["course"] not in half]
    mA = ol.fit_walk(A, ridge=a.ridge, feats=ol.scan_features)
    mB = ol.fit_walk(B, ridge=a.ridge, feats=ol.scan_features)
    train_maps = {**build(A, mB), **build(B, mA)}
    for c in courses:   # a course where the crawl found no jump still gets an (empty) map
        train_maps.setdefault(f"{c}|0.00", [])
    m_all = ol.fit_walk(tr, ridge=a.ridge, feats=ol.scan_features)
    val_maps = build(va, m_all)
    Path("runs/p45-maps-train.json").write_text(json.dumps(train_maps))
    Path("runs/p45-maps-val.json").write_text(json.dumps(val_maps))
    Path("runs/p45-maps-estimator.json").write_text(json.dumps(m_all))
    print(f"train maps: {len(train_maps)} courses, val maps: {len(val_maps)} runs")


if __name__ == "__main__":
    main()
