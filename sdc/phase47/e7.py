"""Phase 4.7 E7: RL fine-tuning of the sensors-only car, with a coach that sees the whole track.

    python -m sdc.phase47.e7 --out runs/p47-e7 [--seed 7] [--hours 9] [--set key=value ...]

E5 and E6 showed the RL racers crash on 11-17% of laps but their sensor copies on 39-48%: most of the loss is in the
copying, not in what the teacher can see. E7 trains the copy itself on its own driving.

  actor   E5's seed-0 student (the owner's GRU network), unchanged inputs: sensors only, its own memory.
  critic  a separate network on the racer's privileged inputs (true state, 30 m of track, next jump, physics),
          started from the RL racer's critic (runs/p47-racer-s1, 3M steps), then warmed on this car's driving.
          Used only in training, to judge how good each moment is ("asymmetric actor-critic").
  reward  as the racer: metres of progress, CRASH_PEN on a crash, FINISH_BONUS for a finished lap.
  PPO     the owner's GAE, clipped policy loss and value loss; the actor is trained on 64-step sequences from the
          memory it actually had (as sdc/phase4/train_rnn.py).
  env     the student's: truggy physics, sensors, steering expo 0.4, training-range physics, E2's train loops.
  score   every eval_every steps: the 150 selection runs, gate range, tiny action noise, 2 passes.
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

from sdc.jumps import env as je
from sdc.phase1 import my_ppo as owner
from sdc.phase4 import my_gru
from sdc.phase4 import train_rnn as trr
from sdc.phase46 import common as cm
from sdc.phase47 import e2_prep
from sdc.phase47 import racer as rc
from sdc.phase47.lap_chaos import Noisy
from sdc.phase47.train_residual import env_kw

CFG = dict(workers=4, cars=32, n_steps=256, n_seq_batch=64, n_epochs=4, lr=3e-5, critic_lr=1e-4, gamma=0.995,
           lam=0.95, clip=0.2, max_grad_norm=0.5, std=0.15, critic_warmup=1_000_000, eval_every=3_000_000,
           crash_pen=200.0, finish_bonus=50.0, actor="runs/p47-e5-s0", critic="runs/p47-racer-s1/ckpt/3014656.pt")
CRASH_PEN, FINISH_BONUS = 200.0, 50.0


def _t(x):
    return torch.as_tensor(np.clip(x, -10, 10), dtype=torch.float32)


# ---- workers: the student's car, returning its sensor inputs and the privileged ones ---------------------------

def _worker(conn, seed, n, courses):
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    b = je.JumpBatch(n, seed=seed, courses=list(rng.permutation(courses)), **env_kw(val=False))

    def reset(c):
        return c.reset(x0=float(rng.uniform(-0.3, 0.3)))

    obs = np.stack([reset(c) for c in b.cars])
    while True:
        cmd, act = conn.recv()
        if cmd == "close":
            return
        if cmd == "obs":
            conn.send((obs, np.stack([rc.priv_obs(c) for c in b.cars])))
            continue
        rew = np.zeros(n, np.float32)
        term, trunc = np.zeros(n, bool), np.zeros(n, bool)
        tpriv = np.zeros((n, rc.OBS_DIM), np.float32)
        eps = []
        for i, c in enumerate(b.cars):
            rew[i], term[i], trunc[i], reason = c.step(act[i], 0.0, b.max_steps, 0.0, FINISH_BONUS)
            if term[i] and reason != "finished":
                rew[i] -= CRASH_PEN
            if term[i] or trunc[i]:
                tpriv[i] = rc.priv_obs(c)
                ep = c.summary(reason)
                eps.append(dict(finished=bool(ep["finished"]), l=int(ep["l"]), reason=reason))
                obs[i] = reset(c)
            else:
                obs[i] = c.obs()
        priv = np.stack([rc.priv_obs(c) for c in b.cars])
        conn.send((obs, priv, rew, term, trunc, tpriv, eps))


class Vec(rc.Vec):
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


# ---- the networks ----------------------------------------------------------------------------------------------

def load_actor(run_or_path, obs_dim):
    net = trr.RecurrentActorCritic(obs_dim=obs_dim)
    p = Path(run_or_path)
    net.load_state_dict(torch.load(p / "model.pt" if p.is_dir() else p))
    return net


def actor_step(net, obs, h, started):
    """The student's actor: memory update and pre-tanh action mean (the critic branch is not used)."""
    a_in, _, h = net.step(obs, h, started)
    return net.actor(a_in), h


def _make_student(path, obs_dim, seed):
    import os
    torch.set_num_threads(1)
    return Noisy(trr.GreedyRNN(load_actor(path, obs_dim)), seed * 1000 + os.getpid())


def score(path, obs_dim, procs=4, passes=2):
    courses = json.loads(e2_prep.OUT.read_text())["split"]["select"]
    fin, laps = [], []
    for r in range(passes):
        eps = cm.prun(functools.partial(_make_student, str(path), obs_dim, r), courses, procs=procs,
                      **env_kw(val=True))
        fin.append(int(sum(bool(e["finished"]) for e in eps)))
        laps += [e["l"] * 0.02 for e in eps if e["finished"]]
    return dict(finished_per_pass=fin, mean=float(np.mean(fin)), lap_s=round(float(np.median(laps)), 2))


# ---- training ----------------------------------------------------------------------------------------------------

def train(out, seed=7, hours=9.0, **over):
    global CRASH_PEN, FINISH_BONUS
    cfg = {**CFG, **over}
    CRASH_PEN, FINISH_BONUS = cfg["crash_pen"], cfg["finish_bonus"]          # read by the forked workers
    out = Path(out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    courses = [int(c) for c in json.loads(e2_prep.OUT.read_text())["split"]["train"]]
    obs_dim = je.JumpBatch(1, **env_kw(val=False)).obs_dim
    net = load_actor(cfg["actor"], obs_dim)
    critic = rc.Net().critic
    rnet = rc.Net()
    rnet.load_state_dict(torch.load(cfg["critic"]))
    critic.load_state_dict(rnet.critic.state_dict())
    with torch.no_grad():
        net.log_std.fill_(float(np.log(cfg["std"])))
    a_params = list(net.cells[0].parameters()) + list(net.actor.parameters()) + [net.log_std]
    opt = torch.optim.Adam([dict(params=a_params, lr=cfg["lr"]),
                            dict(params=critic.parameters(), lr=cfg["critic_lr"])], eps=1e-5)
    log = dict(cfg=cfg, seed=seed, obs_dim=obs_dim, evals=[], updates=[])
    steps, next_eval = 0, 0
    if (out / "resume.pt").exists():
        r = torch.load(out / "resume.pt", weights_only=False)
        net.load_state_dict(r["net"])
        critic.load_state_dict(r["critic"])
        opt.load_state_dict(r["opt"])
        log, steps, next_eval = r["log"], r["steps"], r["next_eval"]
        t0 -= r["wall"]
        print("resumed", steps, flush=True)

    def save():
        torch.save(dict(net=net.state_dict(), critic=critic.state_dict(), opt=opt.state_dict(), log=log, steps=steps,
                        next_eval=next_eval, wall=time.perf_counter() - t0), out / "resume.pt")
        (out / "log.json").write_text(json.dumps(log, default=float))

    vec = Vec(cfg["workers"], cfg["cars"], seed + steps, courses)
    owner.POLICY_LOSS_VARIANT = "textbook"
    L, T, B = my_gru.CONFIG["seq_len"], cfg["n_steps"], vec.n
    assert T % L == 0
    obs, priv = vec.obs()
    h = net.init_memory(B)
    started = torch.ones(B)
    ep_hist = []
    while time.perf_counter() - t0 < hours * 3600:
        if steps >= next_eval:
            p = out / "ckpt" / f"{steps}.pt"
            torch.save(net.state_dict(), p)
            log["evals"].append(dict(steps=steps, wall_s=round(time.perf_counter() - t0),
                                     **score(p, obs_dim, procs=cfg["workers"])))
            print("eval", log["evals"][-1], flush=True)
            next_eval = steps + cfg["eval_every"]
            save()
        buf = {k: [] for k in ("obs", "act", "logp", "rew", "done", "val", "h", "start", "priv")}
        with torch.no_grad():
            for _ in range(T):
                x, xp = _t(obs), _t(priv)
                buf["h"].append(h)
                buf["start"].append(started)
                mu, h = actor_step(net, x, h, started)
                d = torch.distributions.Normal(mu, net.log_std.exp())
                act = d.sample()
                v = critic(xp).squeeze(-1)
                obs, priv, rew, term, trunc, tpriv, eps = vec.step(np.tanh(act.numpy()))
                rew = rew.copy()
                if trunc.any():
                    rew[trunc] += cfg["gamma"] * critic(_t(tpriv[trunc])).squeeze(-1).numpy()
                ep_hist += eps
                done = torch.as_tensor((term | trunc).astype(np.float32))
                for k, val in (("obs", x), ("priv", xp), ("act", act), ("logp", d.log_prob(act).sum(-1)),
                               ("rew", torch.as_tensor(rew)), ("done", done), ("val", v)):
                    buf[k].append(val)
                started = done
                steps += B
            last = critic(_t(priv)).squeeze(-1)
        rew, done, vals = (torch.stack(buf[k]) for k in ("rew", "done", "val"))
        adv = owner.compute_advantages(rew, vals, done, last, cfg["gamma"], cfg["lam"])     # owner
        ret = adv + vals
        O, A, LP, ST, P = (torch.stack(buf[k]) for k in ("obs", "act", "logp", "start", "priv"))
        Hs = torch.stack(buf["h"])                                                         # (T, n_mem, B, H)
        warm = steps <= cfg["critic_warmup"]
        seqs = [(c, i) for c in range(T // L) for i in range(B)]
        for _ in range(cfg["n_epochs"]):
            perm = np.random.permutation(len(seqs))
            for s in range(0, len(seqs), cfg["n_seq_batch"]):
                idx = [seqs[k] for k in perm[s:s + cfg["n_seq_batch"]]]
                cs = torch.as_tensor([c for c, _ in idx])
                ii = torch.as_tensor([i for _, i in idx])
                tt = (cs[:, None] * L + torch.arange(L)[None]).reshape(-1)
                i2 = ii[:, None].expand(-1, L).reshape(-1)
                loss = 0.5 * owner.value_loss(critic(P[tt, i2]).squeeze(-1), ret[tt, i2])           # owner
                if not warm:
                    hh = Hs[cs * L, :, ii].transpose(0, 1)
                    logps = []
                    for t in range(L):
                        tk = cs * L + t
                        mu, hh = actor_step(net, O[tk, ii], hh, ST[tk, ii])
                        d = torch.distributions.Normal(mu, net.log_std.exp())
                        logps.append(d.log_prob(A[tk, ii]).sum(-1))
                    a = adv[tt, i2]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    loss = loss + owner.policy_loss(torch.stack(logps, 1).reshape(-1), LP[tt, i2], a,
                                                    cfg["clip"])                                    # owner
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(a_params + list(critic.parameters()), cfg["max_grad_norm"])
                opt.step()
        with torch.no_grad():
            net.log_std.clamp_(max=float(np.log(cfg["std"])))
        recent = ep_hist[-200:]
        log["updates"].append(dict(steps=steps, wall_s=round(time.perf_counter() - t0), warm=warm,
                                   finished=round(float(np.mean([e["finished"] for e in recent])), 3) if recent else None,
                                   lap_s=round(float(np.median([e["l"] for e in recent if e["finished"]] or [0])) * 0.02, 2),
                                   std=round(float(net.log_std.exp().mean()), 3)))
        ep_hist = ep_hist[-400:]
        if len(log["updates"]) % 5 == 0:
            print("update", log["updates"][-1], flush=True)
            save()
    p = out / "ckpt" / f"{steps}.pt"
    torch.save(net.state_dict(), p)
    log["evals"].append(dict(steps=steps, wall_s=round(time.perf_counter() - t0), **score(p, obs_dim, procs=cfg["workers"])))
    print("eval", log["evals"][-1], flush=True)
    save()
    vec.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hours", type=float, default=9.0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    over = {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    train(a.out, a.seed, a.hours, **over)


if __name__ == "__main__":
    main()
