"""The owner's own PPO (Proximal Policy Optimization). LEARNER-OWNED: see AGENTS.md.

Plumbing (agent): networks, collecting rollouts from the racer, minibatch loop,
optimizer, evaluation and metrics in the same format as train.py so the report works.

Core logic (owner, dictated piece by piece; marked OWNER TODO):
  1. compute_advantages  how much better than expected each action turned out (GAE)
  2. policy_loss         PPO's clipped objective: push up good actions, but not too far
  3. value_loss          teach the critic to predict returns

How one training iteration works:
  a. Drive all cars for n_steps with the current policy, sampling actions
     (mean + noise). Record obs, action, log-probability of that action,
     reward, done, and the critic's value estimate V(obs).
  b. compute_advantages(...) → advantage per step, and return = advantage + value.
  c. For n_epochs, shuffle the steps into minibatches; for each, recompute the
     new log-probabilities and values, and take a gradient step on
     policy_loss + vf_coef · value_loss.

Usage (once the TODOs are done):
    python -m sdc.phase1.my_ppo --seed 0 --steps 300000 --out runs/p1-e5-mine-s0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sdc.racer import evaluate, pursuit, sim

# ---------------------------------------------------------------------------------------
# OWNER TODO 1: advantages
# ---------------------------------------------------------------------------------------


def compute_advantages(rewards, values, dones, last_values, gamma, lam):
    """Advantage of every recorded step.

    Shapes: rewards, values, dones are (T, B): T steps for each of B cars.
      rewards[t, i]  reward car i got for its action at step t
      values[t, i]   critic's estimate V(obs) at step t, before acting
      dones[t, i]    1.0 if car i's episode ended at step t (the next obs is a fresh episode)
      last_values[i] V(obs) after the final step, to value what comes after the rollout
    Returns advantages, shape (T, B).

    Owner's choice (2026-11-08): option c, the blend (GAE), "for the reasons stated in the
    weaknesses of each" of pure Monte Carlo (noisy) and one-step (only as good as the critic).
    Episode ends: "the sum needs to restart in that case - this run isn't punished for past
    failure. But the weights will be."
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_values)  # blended advantage of the step after t, built backwards
    for t in reversed(range(T)):
        next_value = last_values if t == T - 1 else values[t + 1]
        alive = 1.0 - dones[t]  # 0 where the episode ended at t: the next state belongs to a new episode
        # option b, one step: this reward + critic's guess for the next state - critic's guess now.
        # No guess about the next state across an episode end (the sum restarts).
        surprise = rewards[t] + gamma * next_value * alive - values[t]
        # option c: add the later steps' surprises, each faded by gamma*lam per step
        # (lam = 0 -> pure b, lam = 1 -> pure a). Restart the sum at an episode end.
        running = surprise + gamma * lam * alive * running
        advantages[t] = running
    return advantages


# ---------------------------------------------------------------------------------------
# OWNER TODO 2: clipped policy loss
# ---------------------------------------------------------------------------------------


def policy_loss(new_logp, old_logp, advantages, clip):
    """Loss to minimise for the actor, on one minibatch (all 1-D tensors, same length).

    new_logp    log-probability of each recorded action under the policy being trained
    old_logp    log-probability of the same action under the policy that collected it
    advantages  from compute_advantages (normalized per minibatch by the plumbing)
    clip        e.g. 0.2

    Owner's rule (2026-11-08), for an action whose probability ratio has left the ±clip band:
      1. good action already pushed up past 1+clip: stop giving credit
      2. bad action already pushed down past 1-clip: stop
      3. good action accidentally pushed DOWN past 1-clip: still limit (no extra correction)
    So the ratio is held inside the band in every case: once outside, this batch stops moving it.
    (Textbook PPO differs on case 3: it takes min(ratio*adv, clipped*adv), which gives a mistake
    like case 3 the full push back. Set policy_loss="textbook" to compare in E5.)
    """
    ratio = (new_logp - old_logp).exp()
    objective = ratio.clamp(1 - clip, 1 + clip) * advantages
    if POLICY_LOSS_VARIANT == "textbook":
        objective = torch.min(ratio * advantages, objective)
    return -objective.mean()  # minimise the negative = maximise the objective


POLICY_LOSS_VARIANT = "owner"  # "owner" (clip every case) or "textbook" (min with unclipped)


# ---------------------------------------------------------------------------------------
# OWNER TODO 3: value loss
# ---------------------------------------------------------------------------------------


def value_loss(predicted, returns):
    """Loss to minimise for the critic: predicted V(obs) vs. the returns target (1-D tensors)."""
    raise NotImplementedError("OWNER TODO 3")


# ---------------------------------------------------------------------------------------
# Plumbing (agent)
# ---------------------------------------------------------------------------------------


def mlp(n_in, n_out, hidden=(64, 64), out_gain=1.0):
    layers, last = [], n_in
    for h in hidden:
        lin = nn.Linear(last, h)
        nn.init.orthogonal_(lin.weight, np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        last = h
    out = nn.Linear(last, n_out)
    nn.init.orthogonal_(out.weight, out_gain)
    nn.init.zeros_(out.bias)
    return nn.Sequential(*layers, out)


class ActorCritic(nn.Module):
    """Actor: mean steering/throttle from obs, plus a learned spread (std). Critic: V(obs)."""

    def __init__(self, obs_dim=sim.OBS_DIM, act_dim=2):
        super().__init__()
        self.actor = mlp(obs_dim, act_dim, out_gain=0.01)
        self.critic = mlp(obs_dim, 1, out_gain=1.0)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


DEFAULTS = dict(n_envs=8, n_steps=256, batch_size=256, n_epochs=10, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
                vf_coef=0.5, max_grad_norm=0.5, policy_loss="owner")


def train(seed=0, total_timesteps=300_000, out="runs/p1-e5-mine", eval_every=25_000, label="my PPO",
          threads=1, gif=True, **overrides):
    from sdc.phase1.train import ENV_DEFAULTS, _gifs, _summary

    cfg = {**DEFAULTS, **overrides}
    global POLICY_LOSS_VARIANT
    POLICY_LOSS_VARIANT = cfg["policy_loss"]
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = sim.RacerBatch(cfg["n_envs"], seed=seed, **ENV_DEFAULTS)
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], eps=1e-5)
    baseline = evaluate.run_laps(pursuit.PurePursuit())

    def greedy(b):
        with torch.no_grad():
            return net.actor(torch.as_tensor(b.obs())).numpy()

    T, B = cfg["n_steps"], cfg["n_envs"]
    obs = torch.as_tensor(env.reset())
    episodes, evals, steps, next_eval = [], [], 0, 0
    t0 = time.perf_counter()
    while steps < total_timesteps:
        if steps >= next_eval:
            evals.append(dict(timesteps=steps, wall_s=round(time.perf_counter() - t0, 2),
                              **_summary(evaluate.run_laps(greedy))))
            next_eval += eval_every
        # a. rollout
        buf = {k: [] for k in ("obs", "act", "logp", "rew", "done", "val")}
        with torch.no_grad():
            for _ in range(T):
                d = net.dist(obs)
                act = d.sample()
                val = net.value(obs)
                nobs, rew, term, trunc, infos = env.step(act.numpy())
                rew = rew.astype(np.float32).copy()
                for i in np.flatnonzero(trunc):  # time limit: not a real ending, so bootstrap
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
        # b. advantages (owner)
        adv = torch.as_tensor(compute_advantages(rew, val, done, last_val, cfg["gamma"], cfg["lam"]),
                              dtype=torch.float32)
        ret = adv + val
        flat = dict(obs=torch.cat(buf["obs"]), act=torch.cat(buf["act"]), logp=torch.cat(buf["logp"]),
                    adv=adv.reshape(-1), ret=ret.reshape(-1))
        # c. update
        n = T * B
        for _ in range(cfg["n_epochs"]):
            perm = torch.randperm(n)
            for s in range(0, n, cfg["batch_size"]):
                idx = perm[s:s + cfg["batch_size"]]
                a = flat["adv"][idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                d = net.dist(flat["obs"][idx])
                new_logp = d.log_prob(flat["act"][idx]).sum(-1)
                loss = (policy_loss(new_logp, flat["logp"][idx], a, cfg["clip"])
                        + cfg["vf_coef"] * value_loss(net.value(flat["obs"][idx]), flat["ret"][idx]))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
                opt.step()
    evals.append(dict(timesteps=steps, wall_s=round(time.perf_counter() - t0, 2), **_summary(evaluate.run_laps(greedy))))
    train_s = time.perf_counter() - t0
    torch.save(net.state_dict(), out / "model.pt")
    final = evaluate.run_laps(greedy)
    m = dict(label=label, preset="mine", seed=seed, threads=threads, n_envs=cfg["n_envs"], ppo=cfg, env=ENV_DEFAULTS,
             total_timesteps=total_timesteps, trained_timesteps=steps, train_wall_s=round(train_s, 2),
             episodes=episodes, evals=evals, final_eval=final, baseline=baseline,
             compare=evaluate.compare(final, baseline))
    m["gifs"] = _gifs(greedy, out, f"{label}, seed {seed}", final) if gif else []
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--out", default=None)
    p.add_argument("--label", default="my PPO")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = p.parse_args()
    over = {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    m = train(a.seed, a.steps, a.out or f"runs/p1-e5-mine-s{a.seed}", label=a.label, **over)
    c, fe = m["compare"], m["final_eval"]
    print(f"{m['label']}: held-out completion {fe['completion']:.0%}, mean lap {fe['mean_lap_time']} s, "
          f"ratio {c['lap_time_ratio']}, gate {'PASS' if c['gate'] else 'fail'}; {m['train_wall_s']:.0f}s")


if __name__ == "__main__":
    main()
