"""The owner's PPO on the 3D jump course.

Core logic is the owner's, imported unchanged from sdc/phase1/my_ppo.py (learner-owned):
compute_advantages (GAE), policy_loss (textbook clipping), value_loss (squared error).
This file is plumbing (agent): the MuJoCo jump-course batch, tanh-squashed actions, decaying
learning rate, an optional exploration-noise cap, early stopping by validation landing rate,
validation evaluation, GIFs, and metrics.json for the report.

Usage:
    python -m sdc.phase4.train --seed 0 --steps 1000000 --out runs/p4-e1-s0
    python -m sdc.phase4.train ... --set gamma=0.995 --env jump_bonus=20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sdc.jumps import course as crs
from sdc.jumps import driver as dr
from sdc.jumps import env as je
from sdc.jumps import render
from sdc.phase1 import my_ppo as owner

DEFAULTS = dict(n_envs=8, n_steps=256, batch_size=256, n_epochs=10, lr=3e-4, lr_decay=True, gamma=0.99, lam=0.95,
                clip=0.2, vf_coef=0.5, max_grad_norm=0.5, std_cap=None, select=True, init=None, hidden=[64, 64])
ENV = dict(jump_bonus=10.0, max_steps=1500, size=1.0)
OBS_CLIP = 5.0
GIF_COURSES = (10000, 10004)


class ActorCritic(owner.ActorCritic):
    """The owner's Phase 1 network (two 64-unit tanh layers each for actor and critic) on 20 inputs."""

    def __init__(self, obs_dim=je.OBS_DIM, act_dim=2, hidden=(64, 64)):
        nn.Module.__init__(self)
        self.actor = owner.mlp(obs_dim, act_dim, hidden=tuple(hidden), out_gain=0.01)
        self.critic = owner.mlp(obs_dim, 1, hidden=tuple(hidden), out_gain=1.0)
        self.log_std = nn.Parameter(torch.zeros(act_dim))


def _t(obs):
    return torch.as_tensor(np.clip(obs, -OBS_CLIP, OBS_CLIP), dtype=torch.float32)


class Greedy:
    """Deterministic policy (mean action, squashed) for evaluation and GIFs."""

    def __init__(self, net):
        self.net = net

    def __call__(self, b):
        with torch.no_grad():
            return np.tanh(self.net.actor(_t(b.obs())).numpy())


def evaluate(policy, courses=crs.VAL_SEEDS, **env_kw) -> dict:
    eps = dr.run(policy, list(courses), **{k: v for k, v in env_kw.items() if k in ("size", "max_steps")})
    return dict(**dr.score(eps), episodes=eps)


def train(seed=0, total_timesteps=1_000_000, out="runs/p4", eval_every=50_000, label="my PPO", threads=1,
          gif=True, env=None, **overrides) -> dict:
    cfg = {**DEFAULTS, **overrides}
    env_cfg = {**ENV, **(env or {})}
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    batch = je.JumpBatch(cfg["n_envs"], seed=seed, **env_cfg)
    net = ActorCritic(hidden=cfg["hidden"])
    if cfg["init"]:
        net.load_state_dict(torch.load(cfg["init"]))
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], eps=1e-5)
    owner.POLICY_LOSS_VARIANT = "textbook"
    greedy = Greedy(net)

    def val():
        r = evaluate(greedy, size=env_cfg["size"])
        return {k: v for k, v in r.items() if k != "episodes"}

    T, B = cfg["n_steps"], cfg["n_envs"]
    obs = _t(batch.reset())
    episodes, evals, steps, next_eval = [], [], 0, 0
    t0 = time.perf_counter()
    while steps < total_timesteps:
        if steps >= next_eval:
            if cfg["select"]:
                (out / "ckpt").mkdir(exist_ok=True)
                torch.save(net.state_dict(), out / "ckpt" / f"{steps}.pt")
            evals.append(dict(timesteps=steps, wall_s=round(time.perf_counter() - t0, 1), **val()))
            print(f"[{steps:>8}] val landed {evals[-1]['rate']:.0%} finished {evals[-1]['finished']:.0%} "
                  f"({evals[-1]['wall_s']:.0f}s)", flush=True)
            next_eval += eval_every
        frac = min(steps / total_timesteps, 1.0)
        if cfg["std_cap"]:
            cap = cfg["std_cap"][0] + (cfg["std_cap"][1] - cfg["std_cap"][0]) * frac
            with torch.no_grad():
                net.log_std.clamp_(max=float(np.log(cap)))
        if cfg["lr_decay"]:
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * (1 - frac)
        buf = {k: [] for k in ("obs", "act", "logp", "rew", "done", "val")}
        with torch.no_grad():
            for _ in range(T):
                d = net.dist(obs)
                act = d.sample()
                v = net.value(obs)
                nobs, rew, term, trunc, infos = batch.step(np.tanh(act.numpy()))
                rew = rew.astype(np.float32).copy()
                for i in np.flatnonzero(trunc):   # time limit: not a real ending, so bootstrap
                    rew[i] += cfg["gamma"] * float(net.value(_t(infos[i]["terminal_observation"])))
                for i in np.flatnonzero(term | trunc):
                    ep = infos[i]["episode"]
                    episodes.append(dict(timesteps=steps + B, ret=ep["r"], length=ep["l"], landed=ep["landed"],
                                         n_jumps=ep["n_jumps"], finished=ep["finished"], reason=ep["reason"],
                                         max_air=ep["max_air"]))
                buf["obs"].append(obs)
                buf["act"].append(act)
                buf["logp"].append(d.log_prob(act).sum(-1))
                buf["rew"].append(torch.as_tensor(rew))
                buf["done"].append(torch.as_tensor((term | trunc).astype(np.float32)))
                buf["val"].append(v)
                obs = _t(nobs)
                steps += B
            last_val = net.value(obs)
        rew, done, vals = (torch.stack(buf[k]) for k in ("rew", "done", "val"))
        adv = owner.compute_advantages(rew, vals, done, last_val, cfg["gamma"], cfg["lam"])   # owner
        ret = adv + vals
        flat = dict(obs=torch.cat(buf["obs"]), act=torch.cat(buf["act"]), logp=torch.cat(buf["logp"]),
                    adv=adv.reshape(-1), ret=ret.reshape(-1))
        n = T * B
        for _ in range(cfg["n_epochs"]):
            perm = torch.randperm(n)
            for s in range(0, n, cfg["batch_size"]):
                idx = perm[s:s + cfg["batch_size"]]
                a = flat["adv"][idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                d = net.dist(flat["obs"][idx])
                new_logp = d.log_prob(flat["act"][idx]).sum(-1)
                loss = (owner.policy_loss(new_logp, flat["logp"][idx], a, cfg["clip"])                       # owner
                        + cfg["vf_coef"] * owner.value_loss(net.value(flat["obs"][idx]), flat["ret"][idx]))  # owner
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
                opt.step()
    if cfg["select"]:
        torch.save(net.state_dict(), out / "ckpt" / f"{steps}.pt")
    evals.append(dict(timesteps=steps, wall_s=round(time.perf_counter() - t0, 1), **val()))
    print(f"[{steps:>8}] val landed {evals[-1]['rate']:.0%} (final)", flush=True)
    train_s = time.perf_counter() - t0
    torch.save(net.state_dict(), out / "final.pt")
    selection = None
    if cfg["select"]:   # keep the checkpoint with the best validation landing rate (ties: later)
        best = max(evals, key=lambda e: (e["rate"], e["timesteps"]))
        net.load_state_dict(torch.load(out / "ckpt" / f"{best['timesteps']}.pt"))
        selection = dict(chosen=best["timesteps"], rate=best["rate"], final_rate=evals[-1]["rate"])
    torch.save(net.state_dict(), out / "model.pt")
    final = evaluate(greedy, size=env_cfg["size"])
    m = dict(label=label, seed=seed, threads=threads, ppo=cfg, env=env_cfg, total_timesteps=total_timesteps,
             trained_timesteps=steps, train_wall_s=round(train_s, 1), episodes=episodes, evals=evals,
             selection=selection, final={k: v for k, v in final.items() if k != "episodes"},
             final_episodes=final["episodes"])
    if gif:
        m["gifs"] = []
        for c in GIF_COURSES:
            p = out / f"course-{c}.gif"
            ep = render.gif(greedy, c, p, label)
            m["gifs"].append(dict(course=c, path=p.name, landed=ep["landed"], reason=ep["reason"]))
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    return m


def load(run_dir):
    m = json.loads((Path(run_dir) / "metrics.json").read_text())
    net = ActorCritic(hidden=m["ppo"]["hidden"])
    net.load_state_dict(torch.load(Path(run_dir) / "model.pt"))
    return Greedy(net), m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--out", default=None)
    p.add_argument("--label", default="my PPO")
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--no-gif", action="store_true")
    a = p.parse_args()
    parse = lambda kvs: {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in kvs)}  # noqa: E731
    m = train(a.seed, a.steps, a.out or f"runs/p4-s{a.seed}", eval_every=a.eval_every, label=a.label,
              env=parse(a.env), gif=not a.no_gif, **parse(a.set))
    f = m["final"]
    print(f"{m['label']}: val landed {f['rate']:.0%} ({f['landed']}/{f['jumps']}), finished {f['finished']:.0%}; "
          f"trained {m['trained_timesteps']} steps in {m['train_wall_s']:.0f}s")


if __name__ == "__main__":
    main()
