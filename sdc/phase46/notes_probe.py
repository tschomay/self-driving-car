"""Phase 4.6 research probe (not an owner experiment): can the out-lap tell what each jump is?

    python -m sdc.phase46.notes_probe     # -> runs/p46-notes-probe.json

The owner's idea: after a cautious lap a driver knows the track: where the jumps are, what kind, how big. Here the
remote-driven out-lap stand-in (teacher.OutLapDriver) crawls one lap while the car records its parts-list sensors.
From the log alone (no truth): the owner's dead reckoning gives the odometer, the kicker detector (maps.kickers)
marks each jump, and the attitude filter's pitch over the next metres is the jump's profile as the car felt it.

Features per jump: the pitch trace from the kicker mark to 9 m on, resampled every 0.25 m. Labels (for scoring only):
the true jump at that kicker. A nearest-centroid classifier is fit on practice out-laps and scored on validation
out-laps; kicker height is estimated as the climb (integral of sin(pitch) over the odometer while climbing).
"""

from __future__ import annotations

import json
from multiprocessing import Pool

import numpy as np

from sdc.jumps import driver as dr
from sdc.jumps import loop as lp
from sdc.phase46 import common as cm
from sdc.phase46 import laps
from sdc.phase46 import maps
from sdc.phase46 import my_deadreckoning as mdr
from sdc.phase46.teacher import OutLapDriver

DS, SPAN, BACK = 0.25, 9.0, 0.5


def odometer(log):
    o = log["obs"]
    est = mdr.DeadReckoning()
    est.reset(log["still_gyro"])
    for i in range(1, len(o)):
        att = o[i, laps.I_ATT] * 0.5
        est.update(o[i, laps.I_GYRO] * 5.0, o[i, laps.I_SPEED] * 10.0, att[0], att[1], acc=o[i, laps.I_ACC] * 9.81,
                   steer=o[i, laps.I_ACT][0])
    return np.array(est.path)[:, 0]


def jump_features(log):
    """Per kicker mark: (odometer at mark, pitch trace from mark-BACK to mark+SPAN, climb height, true kind/height)."""
    o = log["obs"]
    pitch = o[:, laps.I_ATT.start] * 0.5
    acc = np.linalg.norm(o[:, laps.I_ACC], axis=1)
    odo = odometer(log)
    n = min(len(odo), len(pitch))
    odo, pitch, acc = odo[:n], pitch[:n], acc[:n]
    marks = maps.kickers(pitch, acc, odo)
    course = lp.make(log["course"])
    true_s = log["truth"][:n, 0]
    out = []
    for m in marks:
        i = int(np.searchsorted(odo, m))
        grid = m - BACK + DS * np.arange(int(round((SPAN + BACK) / DS)))
        if grid[-1] > odo[-1]:
            continue
        trace = np.interp(grid, odo, pitch)
        seg = (odo >= m - 2.0) & (odo <= m + 3.0) & (pitch > 0)
        climb = float(np.sum(np.sin(pitch[seg][1:]) * np.diff(odo[seg]))) if seg.sum() > 1 else 0.0
        x = true_s[min(i, n - 1)] % course.length
        j = [jj for jj in course.jumps if jj.start - 2.0 <= x <= jj.lip + 1.0]
        out.append(dict(odo=float(m), true_x=float(x), trace=trace.tolist(), climb=climb,
                        kind=j[0].kind if j else None, height=j[0].height if j else None))
    return out


def _job(args):
    course, x0, split = args
    import torch
    torch.set_num_threads(1)
    log = laps.record(OutLapDriver(), course, x0, dr="gate", phys_seed=cm.PHYS_SEED_VAL, max_steps=6000)
    feats = jump_features(log) if log["reason"] == "finished" else []
    return dict(course=course, x0=x0, split=split, reason=log["reason"], jumps=feats,
                n_true=len(lp.make(course).jumps))


def main():
    jobs = [(c, 0.0, "practice") for c in cm.PRACTICE[:120]] + [(c, x0, "val") for c in cm.VAL_SEEDS for x0 in dr.STARTS]
    with Pool(4) as p:
        laps_ = p.map(_job, jobs, chunksize=1)
    json.dump(laps_, open("runs/p46-notes-probe-laps.json", "w"))
    train = [j for l in laps_ if l["split"] == "practice" for j in l["jumps"] if j["kind"]]
    val = [j for l in laps_ if l["split"] == "val" for j in l["jumps"]]
    kinds = sorted({j["kind"] for j in train})
    cent = {k: np.mean([j["trace"] for j in train if j["kind"] == k], 0) for k in kinds}
    # height: linear fit of true height on the measured climb, on practice
    A = np.array([[j["climb"], 1.0] for j in train])
    coef = np.linalg.lstsq(A, np.array([j["height"] for j in train]), rcond=None)[0]
    conf = {k: {k2: 0 for k2 in kinds + ["missed"]} for k in kinds + ["false mark"]}
    herr = []
    for j in val:
        pred = min(kinds, key=lambda k: float(np.sum((np.array(j["trace"]) - cent[k]) ** 2)))
        conf[j["kind"] or "false mark"][pred] += 1
        if j["kind"]:
            herr.append(abs(coef[0] * j["climb"] + coef[1] - j["height"]))
    val_laps = [l for l in laps_ if l["split"] == "val"]
    found = sum(len([j for j in l["jumps"] if j["kind"]]) for l in val_laps if l["reason"] == "finished")
    total = sum(l["n_true"] for l in val_laps if l["reason"] == "finished")
    res = dict(out_laps_finished={s: sum(l["reason"] == "finished" for l in laps_ if l["split"] == s)
                                  for s in ("practice", "val")},
               val_jumps_marked=found, val_jumps_total=total, confusion=conf,
               accuracy=sum(conf[k][k] for k in kinds) / max(1, sum(conf[k][k2] for k in kinds for k2 in kinds)),
               height_err_median=float(np.median(herr)) if herr else None, height_fit=coef.tolist(),
               centroids={k: v.tolist() for k, v in cent.items()}, laps=laps_)
    json.dump(res, open("runs/p46-notes-probe.json", "w"))
    print({k: v for k, v in res.items() if k not in ("laps", "centroids")})


if __name__ == "__main__":
    main()
