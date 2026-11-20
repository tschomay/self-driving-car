"""Phase 4.7 after E3 (owner: "a map only hurts if it's inaccurate"): what are complete maps worth?

    python -m sdc.phase47.full_maps     -> runs/p47-maps-select-full.json, runs/p47-full-maps.pkl, printed summary
    python -m sdc.phase47.full_maps trainval  -> runs/p47-maps-{val,train}-full.json (E3b's training maps)

10 of the 150 selection runs had no map: their out-lap didn't finish (7 rolled over). A person would simply drive
the out-lap again. Here each failed out-lap is retried with the stand-in's jump speed at 3.5, 3.0, 4.5, then
5.0 m/s (default 4.0) until one finishes. Then the map car (E3's students, unchanged) and the teacher reading the
map drive the loops that had a missing map, 5 noisy passes as in where.py, with the completed maps. Their other
runs keep their E3 results (same maps), so the per-pass totals combine the two.
"""

from __future__ import annotations

import collections
import json
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from sdc.phase46 import laps
from sdc.phase46.maps import make_map
from sdc.phase46.teacher import OUTLAP_TRUGGY, OutLapDriver
from sdc.phase47 import where as W

FULL = Path("runs/p47-maps-select-full.json")
OUT = Path("runs/p47-full-maps.pkl")
LADDER = (3.5, 3.0, 4.5, 5.0)


def _retry(key, d="gate"):
    torch.set_num_threads(1)
    c, x0 = key.split("|")
    for vj in LADDER:
        log = laps.record(OutLapDriver(**{**OUTLAP_TRUGGY, "v_jump": vj}), int(c), float(x0), dr=d,
                          phys_seed=0, max_steps=6000, real="truggy")
        if log["reason"] == "finished":
            return key, dict(make_map(log), v_jump=vj)
    return key, make_map(log)


def _retry_train(key):
    return _retry(key, "train")


def complete(src, dst, d):
    """Copy of the maps in src with every failed out-lap retried (ladder above) at range d; written to dst."""
    maps = json.loads(Path(src).read_text())
    bad = [k for k, v in maps.items() if not v.get("kappa")]
    with Pool(4) as p:
        maps.update(dict(p.map(_retry if d == "gate" else _retry_train, bad, chunksize=1)))
    tmp = Path(dst).with_suffix(".tmp")
    tmp.write_text(json.dumps(maps))
    tmp.replace(dst)
    print(dst, "retried", len(bad), "out-laps;", sum(bool(maps[k].get("kappa")) for k in bad), "now have a map",
          flush=True)


def ep(r):
    return r["ep"] if isinstance(r["ep"], dict) else eval(r["ep"])      # noqa: S307 (our own pickles)


def main():
    import sys
    if sys.argv[1:] == ["trainval"]:      # E3b: complete maps for the student's training and validation runs
        for name, d in (("val", "gate"), ("train", "train")):
            dst = Path(f"runs/p47-maps-{name}-full.json")
            if not dst.exists():
                complete(f"runs/p47-maps-{name}.json", dst, d)
        return
    maps = json.loads(Path(W.SELECT_MAP).read_text())
    bad = [k for k, v in maps.items() if not v.get("kappa")]
    if not FULL.exists():
        with Pool(4) as p:
            maps.update(dict(p.map(_retry, bad, chunksize=1)))
        FULL.write_text(json.dumps(maps))
    maps = json.loads(FULL.read_text())
    print("retried", len(bad), "out-laps;", sum(bool(maps[k].get("kappa")) for k in bad), "now have a map")
    courses = sorted({int(k.split("|")[0]) for k in bad})
    if not OUT.exists():
        W.SELECT_MAP = str(FULL)                       # forked workers inherit it
        W.collect(("map", "mapteacher"), OUT, courses)
    new = pickle.loads(OUT.read_bytes())
    old = pickle.loads(Path("runs/p47-e3-where.pkl").read_bytes())
    for who in ("map", "mapteacher"):
        per = collections.Counter()
        for r in old:
            if r["who"] == who and r["course"] not in courses:
                per[r["rep"]] += ep(r)["finished"]
        fixed = collections.Counter()
        for r in new:
            if r["who"] == who:
                per[r["rep"]] += ep(r)["finished"]
                if f"{r['course']}|{r['start']:.2f}" in bad:
                    fixed[r["rep"]] += ep(r)["finished"]
        print(who, "finished per pass with complete maps:", [per[i] for i in range(W.REPEATS)],
              "mean", np.mean(list(per.values())), "| on the 10 re-mapped runs:", [fixed[i] for i in range(W.REPEATS)])


if __name__ == "__main__":
    main()
