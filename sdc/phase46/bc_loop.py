"""Phase 4.6 E3: DAgger copying of the loop teacher into the owner's GRU network, on loop courses.

    python -m sdc.phase46.bc_loop --out runs/p46-e3-map-s0 --seed 0 --map      # with the out-lap map of the turns
    python -m sdc.phase46.bc_loop --out runs/p46-e3-nomap-s0 --seed 0          # sensors only

The copying itself is Phase 4.5's (sdc/phase45/bc_rnn.py: the teacher labels every step, the student drives from
round 1 on, 64-step chunks from the student's own memory). What differs:
  course    practice loops 100000-100299 (those whose out-lap finished), training-range physics
  student   the owner's GRU + filter inputs, parts-list sensors plus the two angled range sensors; with --map also
            the mapped curvature at 9 distances ahead of the car's own odometer (owner's dead reckoning)
  teacher   teacher.LoopExpert; with --map it plans its corner speeds and steering feed-forward from the car's own
            map (what the student sees), otherwise from the true track
  scoring   validation loops at gate-range physics with their own out-lap maps: laps finished, and finished within
            1.5x the true-state teacher's time (the gate's definition). One start per round, three at the end.
Writes model.pt and metrics.json (load with common.load_student).
"""

from __future__ import annotations

import argparse
import functools
import json
import pickle
import time
from pathlib import Path

import torch

from sdc.jumps.driver import STARTS as dr_starts
from sdc.phase4 import my_gru
from sdc.phase4 import train_rnn as trr
from sdc.phase45 import bc_rnn
from sdc.phase46 import common as cm
from sdc.phase46.teacher import TRUGGY, LoopExpert

TRAIN_MAP, VAL_MAP = "runs/p46-maps-train.json", "runs/p46-maps-val.json"
REF = Path("runs/p46-teacher-val-gate.pkl")     # the true-state teacher's validation laps (for the time limit)


REF_TRUGGY = Path("runs/p46-teacher-val-gate-truggy.pkl")   # the re-tuned teacher's laps on the truggy physics


def student_env(use_map, val=False, real="all", expo=0.0, notes=None, maps=(TRAIN_MAP, VAL_MAP)):
    env = dict(cm.CAR, angled=True, dr="gate" if val else "train")
    if use_map:
        env["turn_map"] = maps[1] if val else maps[0]
    if real != "all":
        env["real"] = real
    if expo:
        env["expo"] = expo
    if notes:
        env["jump_notes"] = notes
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--a-lat", type=float, default=3.5, help="teacher's corner grip budget, m/s² (E5 margin: 2.5)")
    ap.add_argument("--a-brake", type=float, default=None, help="teacher's braking, m/s² (default 3.0; 2.0 with --map)")
    ap.add_argument("--truggy", action="store_true", help="truggy physics and its re-tuned teacher (teacher.TRUGGY)")
    ap.add_argument("--expo", type=float, default=0.0, help="steering expo (env.JumpCar)")
    ap.add_argument("--notes", default=None, help='jump notes inputs: "true" (perfect)')
    ap.add_argument("--train-map", default=TRAIN_MAP, help="out-lap maps of the practice loops (with --map)")
    ap.add_argument("--val-map", default=VAL_MAP, help="out-lap maps of the validation runs (with --map)")
    ap.add_argument("--teacher-racer", default=None, help="Phase 4.7: a privileged RL racer checkpoint as teacher")
    ap.add_argument("--teacher-preview", type=float, default=None, help="Phase 4.7 E6: metres of track the racer sees")
    ap.add_argument("--teacher-json", default=None, help="JSON file of LoopExpert settings over the teacher's defaults")
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    t0 = time.perf_counter()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    maps = json.loads(Path(TRAIN_MAP).read_text())
    courses = [c for c in cm.PRACTICE if maps[f"{c}|0.00"]["kappa"]]
    if a.map and a.train_map != TRAIN_MAP:   # only loops whose own out-lap (on these maps) finished
        own = json.loads(Path(a.train_map).read_text())
        courses = [c for c in courses if own.get(f"{c}|0.00", {}).get("kappa")]
    mp = (a.train_map, a.val_map)
    real = "truggy" if a.truggy else "all"
    env = student_env(a.map, real=real, expo=a.expo, notes=a.notes, maps=mp)
    bc_rnn.ENV.clear()
    bc_rnn.ENV.update(env, steer_cost=0.03, jump_bonus=10.0, courses=courses)
    from sdc.jumps import env as je
    obs_dim = je.JumpBatch(1, **env).obs_dim
    net = trr.RecurrentActorCritic(obs_dim=obs_dim)
    a_brake = a.a_brake if a.a_brake is not None else (2.0 if a.map else 3.0)   # 2.0 with map: brakes earlier
    tkw = dict(TRUGGY) if a.truggy else dict(a_lat=a.a_lat, a_brake=a_brake)
    if a.teacher_json:   # Phase 4.7 E4: the teacher re-tuned for landings
        tkw.update({k: v for k, v in json.loads(Path(a.teacher_json).read_text()).items() if not k.startswith("_")})
    teacher = LoopExpert(use_turn_map=a.map, **tkw)
    if a.teacher_racer:
        from sdc.phase47 import racer as _racer
        _racer.PREVIEW = a.teacher_preview                       # read by forked workers; must match how it was trained
        teacher, tkw = _racer.ExpoDriver(a.teacher_racer), dict(racer=a.teacher_racer, preview=a.teacher_preview)
    ref = pickle.load(open(REF_TRUGGY if a.truggy else REF, "rb"))
    ref1 = ref[::len(dr_starts)]                        # start 0 only (prun orders by course, then start)
    val_env = student_env(a.map, val=True, real=real, expo=a.expo, notes=a.notes, maps=mp)
    L = my_gru.CONFIG["seq_len"]
    data, log = [], []
    for r in range(a.rounds):
        O, A, S, H = bc_rnn.collect(net, teacher, a.steps, a.seed * 1000 + r, use_expert=(r == 0))
        data += bc_rnn.chunks(O, A, S, H, L)
        mse = bc_rnn.fit(net, data, a.epochs, seed=a.seed)
        torch.save(net.state_dict(), out / "model.pt")
        eps = cm.prun(functools.partial(cm.load_student, str(out), obs_dim), cm.VAL_SEEDS, starts=(0.0,),
                      procs=a.procs, **val_env)
        sc = cm.lap_score(eps, teacher=ref1)
        log.append(dict(round=r, samples=len(data) * L, mse=round(mse, 4), finished=sc["finished_n"],
                        in_time=sc["in_time_n"], landed=sc["rate"], wall_s=round(time.perf_counter() - t0)))
        print(log[-1], flush=True)
    eps = cm.prun(functools.partial(cm.load_student, str(out), obs_dim), cm.VAL_SEEDS, procs=a.procs, **val_env)
    final = cm.lap_score(eps, teacher=ref)
    m = dict(label="GRU copy of the loop teacher" + (" + turn map" if a.map else ""), seed=a.seed, rounds=log,
             map=a.map, teacher=tkw, obs_dim=obs_dim, gru=my_gru.CONFIG, env46=val_env, train_env=env, final=final,
             final_episodes=eps, wall_s=round(time.perf_counter() - t0, 1))
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    print(f"final: finished {final['finished_n']}/60, in time {final['in_time_n']}/60, landed {final['rate']:.0%}, "
          f"endings {final['reasons']}")


if __name__ == "__main__":
    main()
