"""Phase 4.7, after E2: where do the coin-flip failures happen, and how does the base car differ from the teacher there?

    python -m sdc.phase47.where collect     -> runs/p47-where.pkl   (base car and teacher, REPEATS noisy passes)
    python -m sdc.phase47.where analyse     -> runs/p47-where.json  (and a printed summary)

Runs: E2's 150 selection runs (50 held-out practice loops x 3 starts, gate-range physics), each driver REPEATS times
with tiny independent action noise (std 0.02, about 1% of full range), as lap_chaos.py. Per run it keeps the episode
summary (per-jump landing results), every takeoff (position, speed, pitch, pitch rate), and a 0.1 s trace (position
along and across the lane, heading error, speed, pitch, curvature, airborne, actions).

Each failure is put in one place:
  landing   ended within LAND_S of touching down off a jump, or in the air
  corner    otherwise, with a turn sharper than 0.1 /m within the last CORNER_M metres
  other     anything else
"""

from __future__ import annotations

import collections
import json
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from sdc.jumps import env as je
from sdc.phase46.ensemble import Ensemble
from sdc.phase46.teacher import TRUGGY, LoopExpert
from sdc.phase47 import e2_prep
from sdc.phase47.lap_chaos import NOISE
from sdc.phase47.train_residual import BASE, env_kw

OUT = Path("runs/p47-where.pkl")
SUMMARY = Path("runs/p47-where.json")
REPEATS = 5
STARTS = (0.0, -0.3, 0.3)
STRIDE = 5
LAND_S, CORNER_M, SHARP = 1.0, 6.0, 0.1
FIELDS = ("t", "x", "y", "yaw", "v", "pitch", "up_z", "kappa", "air", "steer", "throttle")


MAP_CAR = ("runs/p47-e3-s0", "runs/p47-e3-s1")      # E3: the map students (bc_loop --map), averaged
SELECT_MAP = "runs/p47-maps-select.json"
FULL_MAP = "runs/p47-maps-select-full.json"          # after E3: every failed out-lap driven again (full_maps.py)
E3B_CAR = ("runs/p47-e3b-s0", "runs/p47-e3b-s1")    # E3b: map students trained on complete maps
E4_CAR = ("runs/p47-e4-s0", "runs/p47-e4-s1")       # E4: sensors-only students copying the re-tuned teacher
E4_TEACHER = "runs/p47-e4-teacher.json"              # e4_tune.py's chosen settings over TRUGGY
E5_CAR = ("runs/p47-e5-s0", "runs/p47-e5-s1")       # E5: sensors-only students copying the privileged RL racer
E5_RACER = "runs/p47-racer-s1/ckpt/3014656.pt"       # the racer they copy (crash price 200, 3M steps)
E6_CAR = ("runs/p47-e6-s0", "runs/p47-e6-s1")       # E6: sensors-only students copying E6a (racer that sees 4 m)


def _driver(who):
    """(policy, env) for base / teacher / map (E3's map students) / mapteacher (the teacher planning from the map) /
    map_full (E3b's students, complete maps) / teacher_plus (E4's re-tuned teacher) / e4 (E4's students)."""
    env = env_kw(val=True)
    if who in ("map", "mapteacher"):
        env = dict(env, turn_map=SELECT_MAP)
    if who == "map_full":
        return Ensemble(E3B_CAR), dict(env, turn_map=FULL_MAP)
    if who == "e4":
        return Ensemble(E4_CAR), env
    if who == "e5":
        return Ensemble(E5_CAR), env
    if who == "e6":
        return Ensemble(E6_CAR), env
    if who == "racer":                                # privileged, so scored like the teachers: no expo
        from sdc.phase47.racer import Driver
        return Driver(E5_RACER), {k: v for k, v in env.items() if k != "expo"}
    if who.startswith("e4cap"):                       # E4 students with forward throttle scaled to e4capNN percent
        return Capped(Ensemble(E4_CAR), int(who[5:]) / 100), env
    if who in ("teacher", "mapteacher", "teacher_plus"):
        env = {k: v for k, v in env.items() if k != "expo"}
        over = {}
        if who == "teacher_plus":
            over = {k: v for k, v in json.loads(Path(E4_TEACHER).read_text()).items() if not k.startswith("_")}
        return LoopExpert(use_turn_map=who == "mapteacher", **{**TRUGGY, **over}), env
    return Ensemble(MAP_CAR if who == "map" else BASE), env


class Capped:
    """Scales the positive (forward) throttle by k; braking and steering untouched. A crude speed limit."""
    def __init__(self, pol, k):
        self.pol, self.k = pol, k

    def reset(self, n):
        self.pol.reset(n)

    def __call__(self, b):
        a = np.array(self.pol(b), np.float32)
        a[:, 1] = np.where(a[:, 1] > 0, a[:, 1] * self.k, a[:, 1])
        return a


WHO_ID = dict(base=0, teacher=1, map=2, mapteacher=3, map_full=4, teacher_plus=5, e4=6, e4cap95=7, e4cap90=8, racer=9, e5=10, e6=11)


def _drive(who, rep, course):
    torch.set_num_threads(1)
    pol, env = _driver(who)
    rng = np.random.default_rng([rep, course, WHO_ID[who]])
    b = je.JumpBatch(len(STARTS), courses=[course] * len(STARTS), **env)
    b._obs = np.zeros((len(STARTS), b.obs_dim), np.float32)
    for i, c in enumerate(b.cars):
        b._obs[i] = c.reset(course, x0=STARTS[i])
    if hasattr(pol, "reset"):
        pol.reset(len(STARTS))
    n = len(STARTS)
    done = np.zeros(n, bool)
    eps, trace, takeoffs = [None] * n, [[] for _ in range(n)], [[] for _ in range(n)]
    was_air = [False] * n
    while not done.all():
        a = np.asarray(pol(b), np.float32)
        a = np.clip(a + rng.normal(0, NOISE, a.shape), -1, 1).astype(np.float32)
        for i, c in enumerate(b.cars):
            if done[i]:
                continue
            st = c.state()
            if c.airborne and not was_air[i]:
                takeoffs[i].append(dict(t=c.t, x=round(st["x"], 2), v=round(st["v_fwd"], 3),
                                        pitch=round(float(np.degrees(st["pitch"])), 2),
                                        pitch_rate=round(float(np.degrees(st["pitch_rate"])), 1),
                                        jump=c.course.jump_at(st["x"])))
            was_air[i] = c.airborne
            if c.t % STRIDE == 0:
                trace[i].append((c.t, st["x"], st["y"], st["yaw"], st["v_fwd"], st["pitch"], st["up_z"], st["kappa"],
                                 float(c.airborne), float(a[i, 0]), float(a[i, 1])))
        _, _, term, trunc, infos = b.step(a)
        for i in np.flatnonzero((term | trunc) & ~done):
            eps[i] = infos[i]["episode"]
            done[i] = True
    return [dict(who=who, rep=rep, course=course, start=s, ep=eps[i], takeoffs=takeoffs[i],
                 trace=np.array(trace[i], np.float32)) for i, s in enumerate(STARTS)]


def _job(args):
    return _drive(*args)


def collect(whos=("base", "teacher"), out=OUT, courses=None):
    courses = courses or json.loads(e2_prep.OUT.read_text())["split"]["select"]
    jobs = [(who, r, c) for who in whos for r in range(REPEATS) for c in courses]
    with mp.get_context("fork").Pool(4) as pool:
        runs = [r for chunk in pool.imap_unordered(_job, jobs, chunksize=1) for r in chunk]
    Path(out).write_bytes(pickle.dumps(runs))
    print(len(runs), "runs ->", out)


def sharp_turns(c, k=0.15):
    """Track positions (m) where a turn sharper than k /m starts."""
    a = np.abs(c.kappa) > k
    starts = np.flatnonzero(a & ~np.roll(a, 1))
    return [float(i * c.ds) for i in starts]


def place(run, course):
    """landing / corner / other, for a failed run."""
    ep = run["ep"]
    end_x = ep["x"]
    tr = run["trace"]
    if len(tr) and tr[-1][FIELDS.index("air")]:
        return "landing"
    touches = [j for j in ep["jumps"] if j.get("touch") is not None]
    end_t = ep["l"] * 0.02
    if any(0 <= end_t - j["touch"] <= LAND_S for j in touches) or any(j.get("cut_short") for j in ep["jumps"]):
        return "landing"
    s = np.arange(end_x - CORNER_M, end_x + 0.01, 0.25)
    if np.max(np.abs([course.curvature(x % course.length) for x in s])) > SHARP:
        return "corner"
    return "other"


def analyse(src=(OUT,), summary=SUMMARY):
    from sdc.jumps import loop as lp
    runs = [r for f in src for r in pickle.loads(Path(f).read_bytes())]
    whos = sorted({r["who"] for r in runs}, key=lambda w: WHO_ID[w])
    courses = {}

    def course(seed):
        if seed not in courses:
            courses[seed] = lp.make(seed)
        return courses[seed]

    key = lambda r: (r["course"], r["start"])   # noqa: E731
    out = {}
    fin = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in runs:
        fin[r["who"]][key(r)].append(r["ep"]["finished"])
    for who in whos:
        f = {k: np.mean(v) for k, v in fin[who].items()}
        fails = [r for r in runs if r["who"] == who and not r["ep"]["finished"]]
        places = collections.Counter((place(r, course(r["course"])), r["ep"]["reason"]) for r in fails)
        coin = [r for r in fails if 0 < f[key(r)] < 1]
        out[who] = dict(passes=REPEATS, finished_per_pass=[int(sum(r["ep"]["finished"] for r in runs
                                                                 if r["who"] == who and r["rep"] == p))
                                                             for p in range(REPEATS)],
                        always=int(sum(v == 1 for v in f.values())), mixed=int(sum(0 < v < 1 for v in f.values())),
                        never=int(sum(v == 0 for v in f.values())),
                        failures=len(fails), coin_flip_failures=len(coin),
                        where={f"{p} ({why})": n for (p, why), n in places.most_common()},
                        where_coin=dict(collections.Counter(place(r, course(r["course"])) for r in coin)))
    # every jump attempt: landed or not, per driver, per jump instance (course, start, jump index)
    att = collections.defaultdict(lambda: collections.defaultdict(list))
    rows = []
    for r in runs:
        tk = {t["jump"]: t for t in r["takeoffs"] if t["jump"] is not None}
        for j in r["ep"]["jumps"]:
            if j.get("missed"):
                continue
            att[r["who"]][(r["course"], r["start"], j["jump"])].append(bool(j["landed"]))
            t = tk.get(j["jump"], {})
            rows.append(dict(who=r["who"], inst=(r["course"], r["start"], j["jump"]), kind=j["kind"],
                             landed=bool(j["landed"]), v_takeoff=t.get("v"), pitch_takeoff=t.get("pitch"),
                             rate_takeoff=t.get("pitch_rate"), air=j.get("air"), v_touch=j.get("v"),
                             pitch_err=(j["pitch"] - j["slope"]) if j.get("pitch") is not None else None,
                             body=j.get("body"), all4=j.get("all4"), upright=j.get("upright"),
                             design=float(course(r["course"]).jumps[j["jump"]].design_speed)))
    jumps = {}
    for who in whos:
        p = {k: np.mean(v) for k, v in att[who].items()}
        by = collections.defaultdict(list)
        for (c, s, k), v in p.items():
            by[course(c).jumps[k].kind].append(v)
        jumps[who] = dict(instances=len(p), attempts=int(sum(len(v) for v in att[who].values())),
                          landed=round(float(np.mean([x for v in att[who].values() for x in v])), 3),
                          always=int(sum(v == 1 for v in p.values())), mixed=int(sum(0 < v < 1 for v in p.values())),
                          never=int(sum(v == 0 for v in p.values())),
                          by_kind={k: dict(n=len(v), landed=round(float(np.mean(v)), 3),
                                           mixed=int(sum(0 < x < 1 for x in v))) for k, v in sorted(by.items())})
    out["jumps"] = jumps

    def stat(rs, f):
        x = [r[f] for r in rs if r[f] is not None]
        return round(float(np.median(x)), 2) if x else None

    # on the same jump instances: base landed vs base failed vs teacher (all its attempts there)
    base_fail_inst = {r["inst"] for r in rows if r["who"] == "base" and not r["landed"]}
    cmp = {}
    for kind in sorted({r["kind"] for r in rows}):
        sel = [r for r in rows if r["kind"] == kind and r["inst"] in base_fail_inst]
        groups = dict(base_failed=[r for r in sel if r["who"] == "base" and not r["landed"]],
                      base_landed=[r for r in sel if r["who"] == "base" and r["landed"]],
                      teacher=[r for r in sel if r["who"] == "teacher"])
        cmp[kind] = {g: dict(n=len(rs), landed=round(float(np.mean([r["landed"] for r in rs])), 3) if rs else None,
                             v_takeoff=stat(rs, "v_takeoff"), v_minus_design=round(float(np.median(
                                 [r["v_takeoff"] - r["design"] for r in rs if r["v_takeoff"] is not None])), 2)
                             if rs else None,
                             pitch_takeoff=stat(rs, "pitch_takeoff"), rate_takeoff=stat(rs, "rate_takeoff"),
                             air=stat(rs, "air"), pitch_err_touch=stat(rs, "pitch_err"),
                             body_hit=round(float(np.mean([bool(r["body"]) for r in rs])), 2) if rs else None)
                     for g, rs in groups.items()}
    out["at_base_failed_jumps"] = cmp
    # all jumps (every instance): how the two drivers approach
    out["all_jumps"] = {kind: {who: dict(v_minus_design=round(float(np.median(
        [r["v_takeoff"] - r["design"] for r in rows if r["kind"] == kind and r["who"] == who
         and r["v_takeoff"] is not None])), 2),
        v_spread=round(float(np.mean([np.std([r["v_takeoff"] for r in rows if r["inst"] == i and r["who"] == who
                                               and r["v_takeoff"] is not None] or [0])
                                       for i in {r["inst"] for r in rows if r["kind"] == kind}])), 3),
        landed=round(float(np.mean([r["landed"] for r in rows if r["kind"] == kind and r["who"] == who])), 3))
        for who in whos} for kind in sorted({r["kind"] for r in rows})}
    # corner and other failures: speed and lane offset in the last 2 s, vs the teacher at the same place
    tr_t = collections.defaultdict(list)
    for r in runs:
        if r["who"] == "teacher":
            tr_t[(r["course"], r["start"])].append(r["trace"])
    I = {f: i for i, f in enumerate(FIELDS)}
    rows_c = []
    for r in runs:
        if r["who"] != "base" or r["ep"]["finished"] or place(r, course(r["course"])) == "landing":
            continue
        tr = r["trace"]
        end_x = r["ep"]["x"]
        w = tr[tr[:, I["x"]] >= end_x - 8.0]
        if not len(w):
            continue
        tv = [t[(t[:, I["x"]] >= end_x - 8.0) & (t[:, I["x"]] <= end_x)] for t in tr_t[key(r)]]
        tv = [t for t in tv if len(t)]
        rows_c.append(dict(place=place(r, course(r["course"])), reason=r["ep"]["reason"],
                           v_max=float(w[:, I["v"]].max()),
                           v_teacher=float(np.mean([t[:, I["v"]].max() for t in tv])) if tv else None,
                           y_end=float(w[-1, I["y"]]), yaw_max=float(np.degrees(np.abs(w[:, I["yaw"]]).max())),
                           kappa_max=float(np.abs(w[:, I["kappa"]]).max())))
    out["corner_other"] = dict(
        n=len(rows_c),
        v_max_base=round(float(np.median([r["v_max"] for r in rows_c])), 2) if rows_c else None,
        v_max_teacher_same_place=round(float(np.median([r["v_teacher"] for r in rows_c if r["v_teacher"]])), 2)
        if rows_c else None,
        faster_than_teacher=int(sum(r["v_teacher"] is not None and r["v_max"] > r["v_teacher"] + 0.3 for r in rows_c)),
        yaw_max_median=round(float(np.median([r["yaw_max"] for r in rows_c])), 1) if rows_c else None,
        rows=rows_c)
    # every driver: speed through sharp turns (|curvature| > 0.15 /m), mean over the 2 m around each turn's start
    ent = collections.defaultdict(list)
    for r in runs:
        c, tr = course(r["course"]), r["trace"]
        if not len(tr):
            continue
        x = tr[:, I["x"]]
        for s0 in sharp_turns(c):
            for lap_s in (s0, s0 + c.length):
                m = (x >= lap_s - 1.0) & (x <= lap_s + 1.0)
                if m.any():
                    ent[r["who"]].append((r["course"], r["start"], s0, float(tr[m, I["v"]].mean())))
                    break
    out["corner_speed"] = {who: round(float(np.median([e[3] for e in v])), 2) for who, v in ent.items()}
    if "map" in ent or "mapteacher" in ent:
        sel = json.loads(Path(SELECT_MAP).read_text())
        out["select_outlaps_unfinished"] = sum(not m.get("kappa") for m in sel.values())
    summary.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("corner_other", "at_base_failed_jumps")}, indent=1,
                     default=float))
    print(json.dumps({k: v for k, v in out["corner_other"].items() if k != "rows"}, default=float))


def main():
    """collect | analyse, or e3: the map car and the map-following teacher, analysed with the base/teacher runs."""
    cmd = sys.argv[1]
    if cmd == "collect":
        collect()
    elif cmd == "analyse":
        analyse()
    elif cmd == "e3":
        e3 = Path("runs/p47-e3-where.pkl")
        if not e3.exists():
            collect(("map", "mapteacher"), e3)
        analyse((OUT, e3), Path("runs/p47-e3-where.json"))
    elif cmd in ("e3b", "e4teacher", "e4"):    # one more driver, analysed with the base car and the teacher
        who = dict(e3b="map_full", e4teacher="teacher_plus", e4="e4")[cmd]
        pk = Path(f"runs/p47-{cmd}-where.pkl")
        if not pk.exists():
            collect((who,), pk)
        analyse((OUT, pk), Path(f"runs/p47-{cmd}-where.json"))
    elif cmd == "e5":                           # E5: the RL racer and its sensor students
        pk = Path("runs/p47-e5-where.pkl")
        if not pk.exists():
            collect(("racer", "e5"), pk)
        analyse((OUT, pk), Path("runs/p47-e5-where.json"))
    elif cmd == "e6":                           # E6: sensor students of the 4 m racer (E6a)
        pk = Path("runs/p47-e6-where.pkl")
        if not pk.exists():
            collect(("e6",), pk)
        analyse((OUT, pk), Path("runs/p47-e6-where.json"))
    elif cmd == "e4cap":                        # after E4: does a lower speed limit trade the lap time back for finishes?
        pk = Path("runs/p47-e4cap-where.pkl")
        if not pk.exists():
            collect(("e4cap95", "e4cap90"), pk)
        analyse((OUT, pk), Path("runs/p47-e4cap-where.json"))


if __name__ == "__main__":
    main()
