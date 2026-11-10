"""The owner's PPO on the Phase 3 dynamic car.

The core logic is the owner's, unchanged, imported from sdc/phase1/my_ppo.py (learner-owned):
compute_advantages (GAE), policy_loss (textbook clipping, the owner's choice after Phase 1 E5),
value_loss (squared error). This file is plumbing (agent): the dynamic car with its extra
sensors and physics randomization, tanh-squashed actions, a decaying learning rate, and the
Phase 3 evaluation (held-out laps, gate-range laps, robustness sweeps, GIFs, smoothness).

Usage:
    python -m sdc.phase3.my_ppo --seed 0 --steps 500000 --out runs/p3-e7-s0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sdc.phase1 import my_ppo as owner
from sdc.phase3 import robust
from sdc.phase3 import train as p3
from sdc.racer import dynamics as dy

# Same settings as the SB3 runs (Phase 1 preset) plus Phase 3's recipe: decaying LR, squash, accelerometer.
DEFAULTS = dict(n_envs=8, n_steps=256, batch_size=256, n_epochs=10, lr=3e-4, lr_decay=True, gamma=0.99, lam=0.95,
                clip=0.2, vf_coef=0.5, max_grad_norm=0.5, std_cap=None)
# std_cap [start, end]: cap on the exploration noise (std, before tanh), lowered linearly over training
ENV = dict(tracks="random", reward="progress", max_steps=1000, physics="train", dirt=True, realism=2,
           sensors=["accel"], squash=True, history=1)


def train(seed=0, total_timesteps=500_000, out="runs/p3-mine", eval_every=50_000, label="my PPO", threads=1,
          gif=True, sweep=True, env=None, **overrides) -> dict:
    cfg = {**DEFAULTS, **overrides}
    env_cfg = {**ENV, **(env or {})}
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    squash = env_cfg["squash"]
    car = dict(realism=env_cfg["realism"], sensors=tuple(env_cfg["sensors"]), history=env_cfg["history"])
    batch = dy.DynamicRacerBatch(cfg["n_envs"], seed=seed, tracks=env_cfg["tracks"], reward=env_cfg["reward"],
                                 max_steps=env_cfg["max_steps"], physics=env_cfg["physics"], dirt=env_cfg["dirt"], **car)
    net = owner.ActorCritic(obs_dim=batch.obs_dim)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], eps=1e-5)
    owner.POLICY_LOSS_VARIANT = "textbook"

    def greedy(b):
        with torch.no_grad():
            a = net.actor(torch.as_tensor(b.obs())).numpy()
        return np.tanh(a) if squash else a

    T, B = cfg["n_steps"], cfg["n_envs"]
    obs = torch.as_tensor(batch.reset())
    episodes, evals, steps, next_eval = [], [], 0, 0
    t0 = time.perf_counter()
    while steps < total_timesteps:
        if steps >= next_eval:
            evals.append(dict(timesteps=steps, wall_s=round(time.perf_counter() - t0, 2),
                              **p3._summary(robust.laps(greedy, **car)),
                              gate_range=robust.gate_laps(greedy, draws=1, **car)["completion"]))
            next_eval += eval_every
        if cfg["std_cap"]:
            cap = cfg["std_cap"][0] + (cfg["std_cap"][1] - cfg["std_cap"][0]) * min(steps / total_timesteps, 1.0)
            with torch.no_grad():
                net.log_std.clamp_(max=float(np.log(cap)))
        if cfg["lr_decay"]:
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * (1 - steps / total_timesteps)
        buf = {k: [] for k in ("obs", "act", "logp", "rew", "done", "val")}
        with torch.no_grad():
            for _ in range(T):
                d = net.dist(obs)
                act = d.sample()
                val = net.value(obs)
                to_car = np.tanh(act.numpy()) if squash else act.numpy()
                nobs, rew, term, trunc, infos = batch.step(to_car)
                rew = rew.astype(np.float32).copy()
                for i in np.flatnonzero(trunc):   # time limit: not a real ending, so bootstrap
                    rew[i] += cfg["gamma"] * float(net.value(torch.as_tensor(infos[i]["terminal_observation"])))
                for i in np.flatnonzero(term | trunc):
                    ep = infos[i]["episode"]
                    episodes.append(dict(timesteps=steps + B, ret=round(ep["r"], 3), length=ep["l"],
                                         laps=round(infos[i]["laps"], 3), crashed=infos[i]["crashed"],
                                         speed=round(infos[i]["mean_speed"], 2)))
                buf["obs"].append(obs)
                buf["act"].append(act)
                buf["logp"].append(d.log_prob(act).sum(-1))
                buf["rew"].append(torch.as_tensor(rew))
                buf["done"].append(torch.as_tensor((term | trunc).astype(np.float32)))
                buf["val"].append(val)
                obs = torch.as_tensor(nobs)
                steps += B
            last_val = net.value(obs)
        rew, done, val = (torch.stack(buf[k]) for k in ("rew", "done", "val"))
        adv = owner.compute_advantages(rew, val, done, last_val, cfg["gamma"], cfg["lam"])   # owner
        ret = adv + val
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
                loss = (owner.policy_loss(new_logp, flat["logp"][idx], a, cfg["clip"])            # owner
                        + cfg["vf_coef"] * owner.value_loss(net.value(flat["obs"][idx]), flat["ret"][idx]))   # owner
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
                opt.step()
    evals.append(dict(timesteps=steps, wall_s=round(time.perf_counter() - t0, 2),
                      **p3._summary(robust.laps(greedy, **car)),
                      gate_range=robust.gate_laps(greedy, draws=1, **car)["completion"]))
    train_s = time.perf_counter() - t0
    torch.save(net.state_dict(), out / "model.pt")
    m = dict(label=label, preset="mine", seed=seed, threads=threads, n_envs=cfg["n_envs"], ppo=cfg,
             env={**env_cfg, "physics": dy.physics_range(env_cfg["physics"]).to_dict()},
             total_timesteps=total_timesteps, trained_timesteps=steps, train_wall_s=round(train_s, 2),
             episodes=episodes, evals=evals, baseline=p3.baseline_laps(**car))
    return p3.finish(greedy, out, m, gif, sweep)


def load_policy(run_dir):
    """Greedy policy of a saved run (for the gate and GIFs)."""
    m = json.loads((Path(run_dir) / "metrics.json").read_text())
    probe = dy.DynamicRacerBatch(1, **p3.car_of(m["env"]))
    net = owner.ActorCritic(obs_dim=probe.obs_dim)
    net.load_state_dict(torch.load(Path(run_dir) / "model.pt"))
    squash = m["env"]["squash"]

    def greedy(b):
        with torch.no_grad():
            a = net.actor(torch.as_tensor(b.obs())).numpy()
        return np.tanh(a) if squash else a
    return greedy, m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--out", default=None)
    p.add_argument("--label", default="my PPO")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--no-sweep", action="store_true")
    a = p.parse_args()
    parse = lambda kvs: {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in kvs)}  # noqa: E731
    m = train(a.seed, a.steps, a.out or f"runs/p3-mine-s{a.seed}", label=a.label, env=parse(a.env),
              sweep=not a.no_sweep, **parse(a.set))
    c = m["compare"]
    print(f"{m['label']}: nominal held-out {c['completion']:.0%}, gate-range {c['gate_completion']:.0%}, "
          f"lap ratio {c['lap_time_ratio']}; trained {m['trained_timesteps']} steps in {m['train_wall_s']:.0f}s")


if __name__ == "__main__":
    main()
