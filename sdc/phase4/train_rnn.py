"""The owner's PPO with the owner's GRU memory (sdc/phase4/my_gru.py), on the jump course.

Plumbing (agent): a recurrent actor-critic assembled from the owner's CONFIG, rollouts that carry
each car's memory from step to step, and training on sequence chunks: every chunk restarts from the
memory the car actually had at its first step (stored during the rollout) and is unrolled through
the chunk, so gradients flow back up to seq_len steps. PPO's losses are the owner's (Phase 1).

Usage:
    python -m sdc.phase4.train_rnn --seed 0 --steps 1000000 --out runs/p4-e3-gru-s0
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
from sdc.jumps import env as je
from sdc.jumps import render
from sdc.phase1 import my_ppo as owner
from sdc.phase4 import my_gru
from sdc.phase4 import train as tr

DEFAULTS = {**tr.DEFAULTS, "n_seq_batch": 32}   # sequences per minibatch


class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim=je.OBS_DIM, act_dim=2, cfg=None, mlp_hidden=(64, 64)):
        super().__init__()
        cfg = cfg or my_gru.CONFIG
        self.cfg = cfg
        H = cfg["hidden"]
        self.n_mem = 1 if cfg["shared"] else 2
        self.cells = nn.ModuleList([my_gru.GRUCell(obs_dim, H) for _ in range(self.n_mem)])
        head_in = H if cfg["where"] == "before" else H + obs_dim
        self.actor = owner.mlp(head_in, act_dim, hidden=tuple(mlp_hidden), out_gain=0.01)
        self.critic = owner.mlp(head_in, 1, hidden=tuple(mlp_hidden), out_gain=1.0)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.H = H

    def init_memory(self, n):
        return torch.zeros(self.n_mem, n, self.H)

    def step(self, obs, h, started):
        """One step for a batch: reset memory where an episode just started, update it, return heads' inputs."""
        new = []
        for k, cell in enumerate(self.cells):
            hk = my_gru.reset_memory(h[k], started)      # owner
            new.append(cell(obs, hk))                    # owner (gru_step)
        h = torch.stack(new)
        a_mem, c_mem = h[0], h[-1]
        if self.cfg["where"] == "before":
            return a_mem, c_mem, h
        return torch.cat([obs, a_mem], -1), torch.cat([obs, c_mem], -1), h

    def dist_value(self, obs, h, started):
        a_in, c_in, h = self.step(obs, h, started)
        return torch.distributions.Normal(self.actor(a_in), self.log_std.exp()), self.critic(c_in).squeeze(-1), h


class GreedyRNN:
    def __init__(self, net):
        self.net = net
        self.h = None
        self.started = None

    def reset(self, n):
        self.h = self.net.init_memory(n)
        self.started = torch.ones(n)

    def __call__(self, b):
        if self.h is None or self.h.shape[1] != b.n:
            self.reset(b.n)
        with torch.no_grad():
            a_in, _, self.h = self.net.step(tr._t(b.obs()), self.h, self.started)
            self.started = torch.zeros(b.n)
            return np.tanh(self.net.actor(a_in).numpy())


def train(seed=0, total_timesteps=1_000_000, out="runs/p4-rnn", eval_every=50_000, label="my PPO + my GRU",
          threads=1, gif=True, env=None, **overrides) -> dict:
    if not my_gru.ready():
        raise SystemExit("my_gru.py still has OWNER TODOs (gru_step, reset_memory, CONFIG)")
    cfg = {**DEFAULTS, **overrides}
    env_cfg = {**tr.ENV, **(env or {})}
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    batch = je.JumpBatch(cfg["n_envs"], seed=seed, **env_cfg)
    net = RecurrentActorCritic(mlp_hidden=cfg["hidden"])
    if cfg["init"]:
        net.load_state_dict(torch.load(cfg["init"]), strict=False)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], eps=1e-5)
    owner.POLICY_LOSS_VARIANT = "textbook"
    greedy = GreedyRNN(net)
    L = my_gru.CONFIG["seq_len"]

    def val():
        greedy.h = None
        r = tr.evaluate(greedy, size=env_cfg["size"])
        return {k: v for k, v in r.items() if k != "episodes"}

    T, B = cfg["n_steps"], cfg["n_envs"]
    assert T % L == 0, "n_steps must be a multiple of seq_len"
    obs = tr._t(batch.reset())
    h = net.init_memory(B)
    started = torch.ones(B)
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
        buf = {k: [] for k in ("obs", "act", "logp", "rew", "done", "val", "h", "start")}
        with torch.no_grad():
            for _ in range(T):
                buf["h"].append(h)
                buf["start"].append(started)
                d, v, h = net.dist_value(obs, h, started)
                act = d.sample()
                nobs, rew, term, trunc, infos = batch.step(np.tanh(act.numpy()))
                rew = rew.astype(np.float32).copy()
                for i in np.flatnonzero(trunc):   # time limit: bootstrap from the final state, same memory
                    _, tv, _ = net.dist_value(tr._t(infos[i]["terminal_observation"])[None], h[:, i:i + 1],
                                              torch.zeros(1))
                    rew[i] += cfg["gamma"] * float(tv)
                for i in np.flatnonzero(term | trunc):
                    ep = infos[i]["episode"]
                    episodes.append(dict(timesteps=steps + B, ret=ep["r"], length=ep["l"], landed=ep["landed"],
                                         n_jumps=ep["n_jumps"], finished=ep["finished"], reason=ep["reason"],
                                         max_air=ep["max_air"]))
                done = torch.as_tensor((term | trunc).astype(np.float32))
                buf["obs"].append(obs)
                buf["act"].append(act)
                buf["logp"].append(d.log_prob(act).sum(-1))
                buf["rew"].append(torch.as_tensor(rew))
                buf["done"].append(done)
                buf["val"].append(v)
                obs = tr._t(nobs)
                started = done
                steps += B
            _, last_val, _ = net.dist_value(obs, h, started)
        rew, done, vals = (torch.stack(buf[k]) for k in ("rew", "done", "val"))
        adv = owner.compute_advantages(rew, vals, done, last_val, cfg["gamma"], cfg["lam"])   # owner
        ret = adv + vals
        O, A, LP, ST = (torch.stack(buf[k]) for k in ("obs", "act", "logp", "start"))   # (T, B, ...)
        Hs = torch.stack(buf["h"])                                                       # (T, n_mem, B, H)
        # sequences: (chunk c, car i) -> steps c*L .. c*L+L-1
        n_chunks = T // L
        seqs = [(c, i) for c in range(n_chunks) for i in range(B)]
        for _ in range(cfg["n_epochs"]):
            perm = np.random.permutation(len(seqs))
            for s in range(0, len(seqs), cfg["n_seq_batch"]):
                idx = [seqs[k] for k in perm[s:s + cfg["n_seq_batch"]]]
                cs = torch.as_tensor([c for c, _ in idx])
                ii = torch.as_tensor([i for _, i in idx])
                hh = Hs[cs * L, :, ii].transpose(0, 1)        # memory at each chunk's first step (n_mem, n, H)
                logps, values = [], []
                for t in range(L):
                    tt = cs * L + t
                    d, v, hh = net.dist_value(O[tt, ii], hh, ST[tt, ii])
                    logps.append(d.log_prob(A[tt, ii]).sum(-1))
                    values.append(v)
                tt = (cs[:, None] * L + torch.arange(L)[None]).reshape(-1)
                i2 = ii[:, None].expand(-1, L).reshape(-1)
                new_logp = torch.stack(logps, 1).reshape(-1)
                new_val = torch.stack(values, 1).reshape(-1)
                a = adv[tt, i2]
                a = (a - a.mean()) / (a.std() + 1e-8)
                loss = (owner.policy_loss(new_logp, LP[tt, i2], a, cfg["clip"])                 # owner
                        + cfg["vf_coef"] * owner.value_loss(new_val, ret[tt, i2]))             # owner
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
    if cfg["select"]:
        best = max(evals, key=lambda e: (e["rate"], e["timesteps"]))
        net.load_state_dict(torch.load(out / "ckpt" / f"{best['timesteps']}.pt"))
        selection = dict(chosen=best["timesteps"], rate=best["rate"], final_rate=evals[-1]["rate"])
    torch.save(net.state_dict(), out / "model.pt")
    greedy.h = None
    final = tr.evaluate(greedy, size=env_cfg["size"])
    m = dict(label=label, seed=seed, threads=threads, ppo=cfg, gru=my_gru.CONFIG, env=env_cfg,
             total_timesteps=total_timesteps, trained_timesteps=steps, train_wall_s=round(train_s, 1),
             episodes=episodes, evals=evals, selection=selection,
             final={k: v for k, v in final.items() if k != "episodes"}, final_episodes=final["episodes"])
    if gif:
        m["gifs"] = []
        for c in tr.GIF_COURSES:
            p = out / f"course-{c}.gif"
            greedy.h = None
            ep = render.gif(greedy, c, p, label)
            m["gifs"].append(dict(course=c, path=p.name, landed=ep["landed"], reason=ep["reason"]))
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    return m


def load(run_dir):
    m = json.loads((Path(run_dir) / "metrics.json").read_text())
    net = RecurrentActorCritic(cfg=m["gru"], mlp_hidden=m["ppo"]["hidden"])
    net.load_state_dict(torch.load(Path(run_dir) / "model.pt"))
    return GreedyRNN(net), m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--out", default=None)
    p.add_argument("--label", default="my PPO + my GRU")
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--no-gif", action="store_true")
    a = p.parse_args()
    parse = lambda kvs: {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in kvs)}  # noqa: E731
    m = train(a.seed, a.steps, a.out or f"runs/p4-rnn-s{a.seed}", eval_every=a.eval_every, label=a.label,
              env=parse(a.env), gif=not a.no_gif, **parse(a.set))
    f = m["final"]
    print(f"{m['label']}: val landed {f['rate']:.0%} ({f['landed']}/{f['jumps']}), finished {f['finished']:.0%}; "
          f"trained {m['trained_timesteps']} steps in {m['train_wall_s']:.0f}s")


if __name__ == "__main__":
    main()
