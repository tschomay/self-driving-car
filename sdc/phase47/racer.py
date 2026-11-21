"""Phase 4.7, stepping back: a racer trained by reinforcement learning on privileged information (the pilot).

    python -m sdc.phase47.racer --out runs/p47-racer-s0 [--seed 0] [--hours 8]

Every car so far is capped by a hand-coded teacher: students copy it, and small residuals on top of a frozen copy
learned nothing measurable. The standard recipe for agile control (legged robots: Lee et al. 2020; drone racing:
Kaufmann et al. 2023) instead trains a *privileged* policy with RL - it sees the true state, the track ahead and the
physics - and only then distills it into a sensor-only student. This pilot asks the first half: given full
information and a reward for progress, does RL find driving better than the hand-coded teacher on these loops?

  obs     true car state (lane offset, heading error, speeds, attitude, rates, wheel spin, airborne), the track
          ahead (curvature to 30 m, height profile to 30 m), the next jump (distances, kind, design speed), and
          the physics draw (grip, mass, motor). No sensors, no memory.
  reward  metres of progress along the lane each step; CRASH_PEN on a crash; FINISH_BONUS for a finished lap.
          Progress per step rewards speed; the crash penalty and finish bonus reward finishing.
  start   behaviour cloning of E4's re-tuned teacher, with corners at 3.5 m/s^2 (headroom.py), on the same inputs (DAgger: the teacher labels the policy's own
          states), then the critic learns this policy's values before PPO changes it.
  PPO     the owner's GAE, clipped policy loss and value loss (sdc/phase1/my_ppo.py, unchanged), with far bigger
          batches than before (64 cars x 256 steps per update), on 4 worker processes.
  train   the 181 practice loops of E2's train split, training-range physics, one timed lap from a standing start.
  score   every EVAL_EVERY steps: the 150 selection runs (50 loops x 3 starts), gate range, tiny action noise, 2
          passes; the re-tuned teacher finished 142 a pass there.
"""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sdc.jumps import car as cr
from sdc.jumps import env as je
from sdc.phase1 import my_ppo as owner
from sdc.phase46 import common as cm
from sdc.phase46.teacher import TRUGGY, LoopExpert
from sdc.phase47 import e2_prep
from sdc.phase47.lap_chaos import Noisy
from sdc.phase47.train_residual import env_kw

KAP_AHEAD = np.arange(0.0, 30.1, 2.0)                                   # 16 curvature samples
H_AHEAD = np.concatenate([np.arange(0.5, 12.1, 0.5), np.arange(14.0, 30.1, 2.0)])   # 24 + 9 height samples
KINDS = ("double", "drop", "table")
OBS_DIM = 14 + len(KAP_AHEAD) + len(H_AHEAD) + 8 + 3
CRASH_PEN, FINISH_BONUS = 50.0, 50.0
MARGIN = 0.0        # per-step cost for running near the lane edge (|y| > 1.0 m of the 1.5 m half-width)
# E4's re-tuned teacher with 3.5 m/s^2 corners (headroom.py: 2.6 s a lap faster for about the same finishes)
TEACHER = dict(TRUGGY, jump_v={"double": 7.4}, a_lat=3.5, a_brake=3.5)
CFG = dict(workers=4, cars=32, n_steps=256, batch_size=4096, n_epochs=5, lr=1e-4, gamma=0.995, lam=0.95, clip=0.2,
           vf_coef=0.5, max_grad_norm=0.5, std=0.15, hidden=(256, 256), bc_rounds=4, bc_steps=200_000,
           bc_epochs=10, critic_warmup=1_000_000, eval_every=3_000_000, eval0=True, crash_pen=CRASH_PEN, init=None, margin=0.0)


def env_train():
    return {k: v for k, v in env_kw(val=False).items() if k != "expo"}


def env_eval():
    return {k: v for k, v in env_kw(val=True).items() if k != "expo"}


def _next_jump(c, s):
    best = None
    for j in c.jumps:
        d = (j.start - s) % c.length
        if j.start - 2.0 <= s % c.length <= j.end:            # on it now
            d = j.start - s % c.length
        if best is None or d < best[0]:
            best = (d, j)
    return best


def priv_obs(car):
    """The privileged observation: true state, track ahead, next jump, physics."""
    st = car.state()
    c = car.course
    s = st["x"]
    spin = float(car.d.qvel[car.spin_dof].mean()) * cr.WHEEL_R - st["v_fwd"]
    h0 = c.height(s)
    o = [st["y"], st["yaw"], st["v_fwd"] / 10, st["vz"] / 5, st["pitch"], st["roll"], st["up_z"],
         st["pitch_rate"] / 10, st["yaw_rate"] / 5, float(car.airborne), float(car.last_act[0]),
         float(car.last_act[1]), spin / 5, float(getattr(car, "servo_set", 0.0)) / cr.STEER_MAX]
    o += list(c.curvature(s + KAP_AHEAD) * 5)
    o += list((c.height(s + H_AHEAD) - h0) * 2)
    nj = _next_jump(c, s)
    if nj is None:
        o += [3.0, 3.0, 3.0, 0.0, 0, 0, 0, 0.0]
    else:
        d, j = nj
        o += [np.clip(d / 20, -0.5, 3), np.clip((d + j.lip - j.start) / 20, -1, 3),
              np.clip((d + j.land_start - j.start) / 20, -1, 3), j.design_speed / 10]
        o += [float(j.kind == k) for k in KINDS] + [j.height]
    p = car.phys
    o += [p.friction, p.mass, p.motor]
    return np.asarray(o, np.float32)


# ---- worker processes: each drives `cars` cars on the practice loops ----------------------------------------

def _worker(conn, seed, n, courses):
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    b = je.JumpBatch(n, seed=seed, courses=list(rng.permutation(courses)), **env_train())
    teacher = LoopExpert(**TEACHER)

    def reset(c):
        c.reset(x0=float(rng.uniform(-0.3, 0.3)))
        return priv_obs(c)

    obs = np.stack([reset(c) for c in b.cars])
    while True:
        cmd, act = conn.recv()
        if cmd == "close":
            return
        if cmd == "obs":
            conn.send((obs, np.stack([teacher.act(c) for c in b.cars])))
            continue
        rew = np.zeros(n, np.float32)
        term, trunc = np.zeros(n, bool), np.zeros(n, bool)
        term_obs = np.zeros_like(obs)
        eps = []
        for i, c in enumerate(b.cars):
            rew[i], term[i], trunc[i], reason = c.step(act[i], 0.0, b.max_steps, 0.0, FINISH_BONUS)
            if term[i] and reason != "finished":
                rew[i] -= CRASH_PEN
            if MARGIN and not term[i]:
                rew[i] -= MARGIN * (max(0.0, abs(c.state()["y"]) - 1.0) / 0.5) ** 2
            if term[i] or trunc[i]:
                term_obs[i] = priv_obs(c)
                ep = c.summary(reason)
                eps.append(dict(finished=bool(ep["finished"]), l=int(ep["l"]), reason=reason))
                obs[i] = reset(c)
            else:
                obs[i] = priv_obs(c)
        conn.send((obs, np.stack([teacher.act(c) for c in b.cars]), rew, term, trunc, term_obs, eps))


class Vec:
    def __init__(self, workers, cars, seed, courses):
        ctx = mp.get_context("fork")
        self.pipes, self.procs = [], []
        for w in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, seed * 100 + w, cars, courses), daemon=True)
            p.start()
            self.pipes.append(a)
            self.procs.append(p)
        self.n = workers * cars

    def obs(self):
        for p in self.pipes:
            p.send(("obs", None))
        r = [p.recv() for p in self.pipes]
        return np.concatenate([x[0] for x in r]), np.concatenate([x[1] for x in r])

    def step(self, act):
        k = len(act) // len(self.pipes)
        for w, p in enumerate(self.pipes):
            p.send(("step", act[w * k:(w + 1) * k]))
        r = [p.recv() for p in self.pipes]
        cat = [np.concatenate([x[i] for x in r]) for i in range(6)]
        return (*cat, [e for x in r for e in x[6]])

    def close(self):
        for p in self.pipes:
            p.send(("close", None))


# ---- the network ---------------------------------------------------------------------------------------------

class Net(owner.ActorCritic):
    def __init__(self, hidden=(256, 256)):
        nn.Module.__init__(self)
        self.actor = owner.mlp(OBS_DIM, 2, hidden=tuple(hidden), out_gain=0.01)
        self.critic = owner.mlp(OBS_DIM, 1, hidden=tuple(hidden), out_gain=1.0)
        self.log_std = nn.Parameter(torch.zeros(2))


def _t(x):
    return torch.as_tensor(np.clip(x, -10, 10), dtype=torch.float32)


class Driver:
    """Greedy privileged driver for scoring (needs the true state: a teacher, not a car)."""

    def __init__(self, path, hidden=(256, 256)):
        torch.set_num_threads(1)
        self.net = Net(hidden)
        self.net.load_state_dict(torch.load(path))

    def __call__(self, b):
        with torch.no_grad():
            return self.net.actor(_t(np.stack([priv_obs(c) for c in b.cars]))).clamp(-1, 1).numpy()


def _make_driver(path, seed):
    import os
    return Noisy(Driver(path), seed * 1000 + os.getpid())


def _make_teacher(seed):
    import os
    return Noisy(LoopExpert(**TEACHER), seed * 1000 + os.getpid())


def score(path, procs=4, passes=2):
    """path: a checkpoint, or None for the teacher the pilot starts from."""
    courses = json.loads(e2_prep.OUT.read_text())["split"]["select"]
    fin, laps = [], []
    for r in range(passes):
        make = functools.partial(_make_driver, str(path), r) if path else functools.partial(_make_teacher, r)
        eps = cm.prun(make, courses, procs=procs, **env_eval())
        fin.append(int(sum(bool(e["finished"]) for e in eps)))
        laps += [e["l"] * 0.02 for e in eps if e["finished"]]
    return dict(finished_per_pass=fin, mean=float(np.mean(fin)), lap_s=round(float(np.median(laps)), 2))


# ---- training ------------------------------------------------------------------------------------------------

def train(out, seed=0, hours=8.0, **over):
    global CRASH_PEN, MARGIN
    cfg = {**CFG, **over}
    CRASH_PEN, MARGIN = cfg["crash_pen"], cfg["margin"]                 # read by the forked workers
    out = Path(out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    courses = [int(c) for c in json.loads(e2_prep.OUT.read_text())["split"]["train"]]
    net = Net(cfg["hidden"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], eps=1e-5)
    log = dict(cfg=cfg, seed=seed, bc=[], evals=[], updates=[])
    steps, next_eval, phase = 0, 0 if cfg["eval0"] else cfg["eval_every"], "bc"
    if (out / "resume.pt").exists():
        r = torch.load(out / "resume.pt", weights_only=False)
        net.load_state_dict(r["net"])
        opt.load_state_dict(r["opt"])
        log, steps, next_eval, phase = r["log"], r["steps"], r["next_eval"], r["phase"]
        t0 -= r["wall"]
        print("resumed", phase, steps, flush=True)
    elif cfg["init"]:                            # start PPO from a checkpoint (critic re-warms: rewards changed)
        net.load_state_dict(torch.load(cfg["init"]))
        phase = "ppo"
        print("init", cfg["init"], flush=True)

    def save():
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), log=log, steps=steps, next_eval=next_eval,
                        phase=phase, wall=time.perf_counter() - t0), out / "resume.pt")
        (out / "log.json").write_text(json.dumps(log, default=float))

    vec = Vec(cfg["workers"], cfg["cars"], seed + steps, courses)
    rng = np.random.default_rng(seed + steps)

    # 1. behaviour cloning (DAgger): round 0 the teacher drives (with noise), later rounds the policy drives
    if phase == "bc":
        X, Y = [], []
        for rnd in range(cfg["bc_rounds"]):
            obs, lab = vec.obs()
            n = 0
            fin = []
            while n < cfg["bc_steps"]:
                with torch.no_grad():
                    a = lab if rnd == 0 else net.actor(_t(obs)).numpy()
                a = np.clip(a + rng.normal(0, 0.1, a.shape), -1, 1).astype(np.float32)
                X.append(obs)
                Y.append(lab)
                obs, lab, _, _, _, _, eps = vec.step(a)
                fin += [e["finished"] for e in eps]
                n += vec.n
            xs, ys = _t(np.concatenate(X)), torch.as_tensor(np.concatenate(Y))
            bopt = torch.optim.Adam(net.actor.parameters(), lr=1e-3)
            for _ in range(cfg["bc_epochs"]):
                perm = torch.randperm(len(xs))
                for i in range(0, len(xs), 1024):
                    idx = perm[i:i + 1024]
                    loss = ((net.actor(xs[idx]) - ys[idx]) ** 2).mean()
                    bopt.zero_grad()
                    loss.backward()
                    bopt.step()
            log["bc"].append(dict(round=rnd, samples=len(xs), mse=round(float(loss), 4),
                                  finished=round(float(np.mean(fin)), 3) if fin else None,
                                  wall_s=round(time.perf_counter() - t0)))
            print("bc", log["bc"][-1], flush=True)
        with torch.no_grad():
            net.log_std.fill_(float(np.log(cfg["std"])))
        phase = "ppo"
        torch.save(net.state_dict(), out / "ckpt" / "bc.pt")
        save()

    # 2. PPO (the first critic_warmup steps train only the critic, so it knows this policy's values first)
    owner.POLICY_LOSS_VARIANT = "textbook"
    obs, _ = vec.obs()
    ep_hist = []
    while time.perf_counter() - t0 < hours * 3600:
        if steps >= next_eval:
            p = out / "ckpt" / f"{steps}.pt"
            torch.save(net.state_dict(), p)
            sc = score(p, procs=cfg["workers"])
            log["evals"].append(dict(steps=steps, wall_s=round(time.perf_counter() - t0), **sc))
            print("eval", log["evals"][-1], flush=True)
            next_eval = steps + cfg["eval_every"]
            save()
        buf = {k: [] for k in ("obs", "act", "logp", "rew", "done", "val")}
        with torch.no_grad():
            for _ in range(cfg["n_steps"]):
                x = _t(obs)
                d = net.dist(x)
                act = d.sample()
                v = net.value(x)
                obs, _, rew, term, trunc, tobs, eps = vec.step(act.numpy())
                rew = rew.copy()
                if trunc.any():
                    rew[trunc] += cfg["gamma"] * net.value(_t(tobs[trunc])).numpy()
                ep_hist += eps
                buf["obs"].append(x)
                buf["act"].append(act)
                buf["logp"].append(d.log_prob(act).sum(-1))
                buf["rew"].append(torch.as_tensor(rew))
                buf["done"].append(torch.as_tensor((term | trunc).astype(np.float32)))
                buf["val"].append(v)
                steps += vec.n
            last = net.value(_t(obs))
        rew, done, vals = (torch.stack(buf[k]) for k in ("rew", "done", "val"))
        adv = owner.compute_advantages(rew, vals, done, last, cfg["gamma"], cfg["lam"])     # owner
        ret = adv + vals
        flat = dict(obs=torch.cat(buf["obs"]), act=torch.cat(buf["act"]), logp=torch.cat(buf["logp"]),
                    adv=adv.reshape(-1), ret=ret.reshape(-1))
        warm = steps <= cfg["critic_warmup"]
        n = len(flat["adv"])
        for _ in range(cfg["n_epochs"]):
            perm = torch.randperm(n)
            for i in range(0, n, cfg["batch_size"]):
                idx = perm[i:i + cfg["batch_size"]]
                a = flat["adv"][idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                d = net.dist(flat["obs"][idx])
                loss = cfg["vf_coef"] * owner.value_loss(net.value(flat["obs"][idx]), flat["ret"][idx])   # owner
                if not warm:
                    loss = loss + owner.policy_loss(d.log_prob(flat["act"][idx]).sum(-1), flat["logp"][idx], a,
                                                    cfg["clip"])                                          # owner
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
                opt.step()
        with torch.no_grad():
            net.log_std.clamp_(max=float(np.log(cfg["std"])))
        recent = ep_hist[-200:]
        log["updates"].append(dict(steps=steps, wall_s=round(time.perf_counter() - t0), warm=warm,
                                   finished=round(float(np.mean([e["finished"] for e in recent])), 3) if recent else None,
                                   lap_s=round(float(np.median([e["l"] for e in recent if e["finished"]] or [0])) * 0.02, 2),
                                   std=round(float(net.log_std.exp().mean()), 3)))
        ep_hist = ep_hist[-400:]
        if len(log["updates"]) % 10 == 0:
            print("update", log["updates"][-1], flush=True)
            save()
    p = out / "ckpt" / f"{steps}.pt"
    torch.save(net.state_dict(), p)
    log["evals"].append(dict(steps=steps, wall_s=round(time.perf_counter() - t0), **score(p, procs=cfg["workers"])))
    print("eval", log["evals"][-1], flush=True)
    save()
    vec.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--score-teacher", action="store_true", help="score the starting teacher on the selection runs")
    a = ap.parse_args()
    if a.score_teacher:
        sc = score(None)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "teacher.json").write_text(json.dumps(sc))
        print("teacher", sc, flush=True)
        return
    over = {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    train(a.out, a.seed, a.hours, **over)


if __name__ == "__main__":
    main()
