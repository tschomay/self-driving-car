"""Lap evaluation on held-out tracks: the Phase 1 gate metric.

Each track gets one attempt: start at the start line from standstill, drive one
lap. Outcome per track: finished (with lap time), crashed (off the road), or
timed out (MAX_LAP_S). A policy is any callable policy(batch) -> actions (B, 2);
RL policies read batch.obs(), the pure-pursuit baseline reads the track directly.
"""

from __future__ import annotations

import numpy as np

from sdc.racer import sim
from sdc.racer import track as trk

MAX_LAP_S = 120.0


def run_laps(policy, seeds=trk.HELD_OUT_SEEDS, record: bool = False, make_batch=None, setup=None) -> dict:
    """make_batch(**kw) builds the batch (default sim.RacerBatch); setup(b) runs after reset (e.g. set physics)."""
    seeds = list(seeds)
    make_batch = make_batch or sim.RacerBatch
    b = make_batch(len(seeds), tracks=seeds, laps=1, start="line", max_steps=int(MAX_LAP_S / sim.DT))
    b.reset()
    if setup is not None:
        setup(b)
    n = b.n
    active = np.ones(n, bool)
    rows = [None] * n
    traj = [[] for _ in range(n)] if record else None

    def snap(i, a):
        traj[i].append(dict(pos=b.pos[i].copy(), heading=float(b.heading[i]), v=float(b.v[i]),
                            rays=b.rays[i].copy(), steer=float(a[i, 0]), throttle=float(a[i, 1]),
                            progress=float(b.progress[i] / b.length[i]), t=int(b.steps[i])))
        if hasattr(b, "slip"):   # dynamic car (Phase 3)
            traj[i][-1]["slip"] = b.slip[i].copy()
            traj[i][-1]["dirt"] = bool(b.dirt_pts[i, b.idx[i]])
        imgs = getattr(policy, "last_images", None)   # camera policies: what the car saw this step
        if imgs is not None:
            traj[i][-1]["image"] = imgs[i].copy()

    while active.any():
        a = np.asarray(policy(b), dtype=float)
        if record:
            for i in np.flatnonzero(active):
                snap(i, a)
        _, _, term, trunc, infos = b.step(a)
        for i in np.flatnonzero(active & (term | trunc)):
            info = infos[i]
            finished = not info["crashed"] and info["laps"] >= 1.0 - 1e-6
            rows[i] = dict(track=seeds[i], finished=bool(finished), crashed=bool(info["crashed"]),
                           **({"physics": info["physics"]} if "physics" in info else {}),
                           lap_time=round(info["episode"]["l"] * sim.DT, 2) if finished else None,
                           laps=round(min(info["laps"], 1.0), 3), steps=info["episode"]["l"])
            if record:
                fs = info["final_state"]
                traj[i].append(dict(pos=fs["pos"], heading=fs["heading"], v=fs["v"], rays=fs["rays"],
                                    steer=float(a[i, 0]), throttle=float(a[i, 1]),
                                    progress=rows[i]["laps"], t=info["episode"]["l"]))
                for k in ("slip", "dirt"):
                    if k in traj[i][-2]:
                        traj[i][-1][k] = fs.get(k, traj[i][-2][k])
                if "image" in traj[i][-2]:
                    traj[i][-1]["image"] = traj[i][-2]["image"]
            active[i] = False
    out = dict(tracks=rows, completion=float(np.mean([r["finished"] for r in rows])))
    times = [r["lap_time"] for r in rows if r["finished"]]
    out["mean_lap_time"] = float(np.mean(times)) if times else None
    if record:
        out["trajectories"] = traj
    return out


def compare(rl: dict, baseline: dict) -> dict:
    """Lap-time ratio RL / baseline over tracks both finished, plus the gate verdict."""
    pairs = [(a["lap_time"], b["lap_time"]) for a, b in zip(rl["tracks"], baseline["tracks"])
             if a["finished"] and b["finished"]]
    ratio = float(np.mean([a for a, _ in pairs]) / np.mean([b for _, b in pairs])) if pairs else None
    return dict(completion=rl["completion"], lap_time_ratio=ratio,
                gate=bool(rl["completion"] >= 0.95 and ratio is not None and ratio <= 1.1))
