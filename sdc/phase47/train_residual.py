"""Phase 4.7: the owner's PPO learns a residual correction on top of the frozen base car, in drills.

    python -m sdc.phase47.train_residual --seed 0 --out runs/p47-e1-s0

Core logic is the owner's, imported unchanged from sdc/phase1/my_ppo.py (learner-owned): compute_advantages (GAE),
policy_loss (textbook clipping), value_loss (squared error), and the network (owner.mlp, via phase4 train.ActorCritic).
This file is plumbing (agent), following sdc/phase4/train.py: drills env, exploration-noise cap, critic warm-up,
checkpoints scored on 60 validation runs (validation loops x 3 starts, gate range), resume after restarts.

The residual's last layer starts at exactly zero, so at step 0 the car drives exactly as the base car does.
"""

from __future__ import annotations

import argparse
import functools
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sdc.phase1 import my_ppo as owner
from sdc.phase4 import train as tr
from sdc.phase46 import common as cm
from sdc.phase46.bc_loop import REF_TRUGGY, TRAIN_MAP, student_env
from sdc.phase47 import drills as dl

BASE = ("runs/p46-e10-b-s0", "runs/p46-e10-b-s1")
CFG = dict(n_envs=8, n_steps=256, batch_size=256, n_epochs=10, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, vf_coef=0.5,
           max_grad_norm=0.5, std=(0.5, 0.17), critic_warmup=50_000, harvest_laps=80)


def env_kw(val=False):
    return student_env(False, val=val, real="truggy", expo=0.4)


def validate(base, actor_path, obs_dim, procs=1):
    eps = cm.prun(functools.partial(dl.load_residual, tuple(base), str(actor_path), obs_dim), cm.VAL_SEEDS,
                  procs=procs, **env_kw(val=True))
    sc = cm.lap_score(eps, teacher=pickle.load(open(REF_TRUGGY, "rb")))
    return sc, eps


def train(seed, steps, out, eval_every, base=BASE, procs=1, **over):
    cfg = {**CFG, **over}
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    out = Path(out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    maps = json.loads(Path(TRAIN_MAP).read_text())
    practice = [c for c in cm.PRACTICE if maps[f"{c}|0.00"]["kappa"]]
    laps = list(np.random.default_rng(seed).choice(practice, cfg["harvest_laps"], replace=False))
    pool, hv = dl.harvest(base, [int(c) for c in laps], env_kw(), seed=seed)
    hv["wall_s"] = round(time.perf_counter() - t0)
    print("harvest", hv, flush=True)
    env = dl.DrillEnv(pool, base, cfg["n_envs"], env_kw(), seed=seed)
    net = tr.ActorCritic(obs_dim=env.obs_dim)
    with torch.no_grad():   # zero correction at the start: the base car exactly
        net.actor[-1].weight.zero_()
        net.actor[-1].bias.zero_()
        net.log_std.fill_(float(np.log(cfg["std"][0])))
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], eps=1e-5)
    owner.POLICY_LOSS_VARIANT = "textbook"
    n_steps, evals, drills, done_steps, next_eval = cfg["n_steps"], [], [], 0, eval_every
    if (out / "resume.pt").exists():
        r = torch.load(out / "resume.pt", weights_only=False)
        net.load_state_dict(r["net"])
        opt.load_state_dict(r["opt"])
        done_steps, next_eval, evals, drills = r["steps"], r["next_eval"], r["evals"], r["drills"]
        t0 -= r["wall"]
        torch.manual_seed(seed + done_steps)
        env.rng = np.random.default_rng(seed + done_steps)
        print(f"resumed at {done_steps}", flush=True)
    obs = torch.as_tensor(env.reset())
    B = cfg["n_envs"]
    while done_steps < steps:
        frac = min(done_steps / steps, 1.0)
        cap = cfg["std"][0] + (cfg["std"][1] - cfg["std"][0]) * frac
        with torch.no_grad():
            net.log_std.clamp_(max=float(np.log(cap)))
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * (1 - frac)
        buf = {k: [] for k in ("obs", "act", "logp", "rew", "done", "val")}
        with torch.no_grad():
            for _ in range(n_steps):
                x = tr._t(obs.numpy())
                d = net.dist(x)
                act = d.sample()
                v = net.value(x)
                nobs, rew, term, trunc, infos = env.step(act.numpy())
                rew = rew.copy()
                for i in np.flatnonzero(trunc):
                    rew[i] += cfg["gamma"] * float(net.value(tr._t(infos[i]["terminal_observation"][None]))[0])
                for i in np.flatnonzero(term | trunc):
                    drills.append(dict(steps=done_steps, kind=infos[i]["kind"], reason=infos[i]["reason"],
                                       ret=round(float(infos[i]["ret"]), 2)))
                buf["obs"].append(x)
                buf["act"].append(act)
                buf["logp"].append(d.log_prob(act).sum(-1))
                buf["rew"].append(torch.as_tensor(rew))
                buf["done"].append(torch.as_tensor((term | trunc).astype(np.float32)))
                buf["val"].append(v)
                obs = torch.as_tensor(nobs)
                done_steps += B
            last_val = net.value(tr._t(obs.numpy()))
        rew, done, vals = (torch.stack(buf[k]) for k in ("rew", "done", "val"))
        adv = owner.compute_advantages(rew, vals, done, last_val, cfg["gamma"], cfg["lam"])   # owner
        ret = adv + vals
        flat = dict(obs=torch.cat(buf["obs"]), act=torch.cat(buf["act"]), logp=torch.cat(buf["logp"]),
                    adv=adv.reshape(-1), ret=ret.reshape(-1))
        warm = done_steps <= cfg["critic_warmup"]
        n = n_steps * B
        for _ in range(cfg["n_epochs"]):
            perm = torch.randperm(n)
            for s in range(0, n, cfg["batch_size"]):
                idx = perm[s:s + cfg["batch_size"]]
                a = flat["adv"][idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                d = net.dist(flat["obs"][idx])
                new_logp = d.log_prob(flat["act"][idx]).sum(-1)
                loss = cfg["vf_coef"] * owner.value_loss(net.value(flat["obs"][idx]), flat["ret"][idx])   # owner
                if not warm:
                    loss = loss + owner.policy_loss(new_logp, flat["logp"][idx], a, cfg["clip"])          # owner
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
                opt.step()
        if done_steps >= next_eval or done_steps >= steps:
            p = out / "ckpt" / f"{done_steps}.pt"
            torch.save(net.state_dict(), p)
            sc, _ = validate(base, p, env.obs_dim, procs)
            recent = drills[-2000:]
            crash = {k: round(np.mean([r["reason"] in dl.CRASHES for r in recent if r["kind"] == k]), 3)
                     for k in env.kinds if any(r["kind"] == k for r in recent)}
            evals.append(dict(steps=done_steps, wall_s=round(time.perf_counter() - t0), finished=sc["finished_n"],
                              in_time=sc["in_time_n"], lap_s=sc["lap_s"], landed=sc["rate"], reasons=sc["reasons"],
                              drill_crash=crash, std=round(float(net.log_std.exp().mean()), 3)))
            print(evals[-1], flush=True)
            next_eval += eval_every
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), steps=done_steps, next_eval=next_eval,
                            evals=evals, drills=drills, wall=time.perf_counter() - t0), out / "resume.pt")
    best = max(evals, key=lambda e: (e["in_time"], -e["lap_s"], e["steps"]))
    sel = out / "ckpt" / f"{best['steps']}.pt"
    torch.save(torch.load(sel), out / "model.pt")
    m = dict(label="residual PPO on drills", seed=seed, cfg=cfg, base=list(base), harvest=hv, obs_dim=env.obs_dim,
             scale=dl.DrillEnv.SCALE, crash_pen=dl.DrillEnv.CRASH_PEN, res_cost=dl.DrillEnv.RES_COST,
             mix=dl.DrillEnv.MIX, evals=evals, chosen=best, drills=drills, wall_s=round(time.perf_counter() - t0))
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    print("chosen", best, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--eval-every", type=int, default=200_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    over = {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    train(a.seed, a.steps, a.out, a.eval_every, procs=a.procs, **over)


if __name__ == "__main__":
    main()
