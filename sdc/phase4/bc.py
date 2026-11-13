"""Phase 4 E4: behaviour cloning of the hand-coded expert into the owner's network, as a PPO starting point.

    python -m sdc.phase4.bc --out runs/p4-e4-bc-s0 --seed 0

The expert (ExpertDriver at a steady 7.1 m/s, 95% on validation) reads the true state: pitch, speed,
lane position, the terrain where it will land. The student is the owner's Phase 1 network on the real
sensors only (E3's inputs: IMU, wheel speeds, forward rays, heading fan, steering angle, last action),
so it has to work out from its sensors what the expert reads directly. Same recipe as Phase 3.5 E4:
  - actor: DAgger. Round 0 drives with the expert; later rounds drive with the student (plus a little
    noise) and the expert labels every state it visits. Loss: squared error between tanh(actor mean)
    and the expert's action.
  - critic: regressed on the discounted return (gamma 0.997, the PPO reward) of the student's own
    noisy driving, so PPO's first updates see sensible advantages.
Writes model.pt (a state_dict for train.py --set init=...), metrics.json and the student's scores.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sdc.jumps import course as crs
from sdc.jumps import driver as dr
from sdc.jumps import env as je
from sdc.phase4 import train as tr

ENV = dict(fan=True, servo=True, steer_cost=0.03, jump_bonus=10.0, max_steps=1500)
GAMMA = 0.997
EXPERT = dict(plan=False, v_target=7.1)


def batch(seed, n=16):
    return je.JumpBatch(n, seed=seed, **ENV)


def collect(net, expert, n_steps, seed, use_expert, noise=0.1):
    """States the driver (expert or student) visits, the expert's action there, rewards and dones."""
    b = batch(seed)
    b.reset()
    rng = np.random.default_rng(seed)
    stud = tr.Greedy(net)
    O, A, R, D = [], [], [], []
    for _ in range(n_steps // b.n):
        o = b.obs().copy()
        a_exp = expert(b)              # the expert labels the state (it also tracks takeoff throttle per car)
        a = a_exp if use_expert else stud(b)
        a = np.clip(a + noise * rng.standard_normal(a.shape), -1, 1)
        _, r, term, trunc, _ = b.step(a)
        O.append(o)
        A.append(a_exp.astype(np.float32))
        R.append(r.astype(np.float32))
        D.append(term | trunc)
    return np.stack(O), np.stack(A), np.stack(R), np.stack(D)


def returns(R, D):
    G = np.zeros_like(R)
    nxt = np.zeros(R.shape[1], np.float32)
    for t in range(len(R) - 1, -1, -1):
        nxt = R[t] + GAMMA * nxt * (~D[t])
        G[t] = nxt
    return G


def fit(module, X, Y, epochs, lr=1e-3, squash=False, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    X = torch.as_tensor(np.clip(X, -tr.OBS_CLIP, tr.OBS_CLIP))
    Y = torch.as_tensor(Y)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for s in range(0, len(X), 512):
            i = perm[s:s + 512]
            out = module(X[i])
            out = torch.tanh(out) if squash else out.squeeze(-1)
            loss = ((out - Y[i]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        out = module(X)
        out = torch.tanh(out) if squash else out.squeeze(-1)
        return float(((out - Y) ** 2).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50_000)
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    t0 = time.perf_counter()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    expert = dr.ExpertDriver(**EXPERT)
    net = tr.ActorCritic(obs_dim=batch(0, 1).obs_dim)
    env_kw = dict(fan=ENV["fan"], servo=ENV["servo"])
    Xs, Ys, log = [], [], []
    for r in range(a.rounds):
        O, A, _, _ = collect(net, expert, a.steps, a.seed * 1000 + r, use_expert=(r == 0))
        Xs.append(O.reshape(-1, O.shape[-1]))
        Ys.append(A.reshape(-1, 2))
        mse = fit(net.actor, np.concatenate(Xs), np.concatenate(Ys), epochs=15, squash=True, seed=a.seed)
        sc = dr.score(dr.run(tr.Greedy(net, env_kw), crs.VAL_SEEDS))
        log.append(dict(round=r, samples=int(sum(len(x) for x in Xs)), mse=round(mse, 4), rate=sc["rate"],
                        finished=sc["finished"]))
        print(log[-1], flush=True)
    O, _, R, D = collect(net, expert, 2 * a.steps, a.seed * 1000 + 99, use_expert=False, noise=0.15)
    G = returns(R, D)
    vmse = fit(net.critic, O.reshape(-1, O.shape[-1]), G.reshape(-1), epochs=20, seed=a.seed)
    print(dict(critic_mse=round(vmse, 2), mean_return=round(float(G.mean()), 1)), flush=True)
    torch.save(net.state_dict(), out / "model.pt")
    final = tr.evaluate(tr.Greedy(net, env_kw), starts=dr.STARTS, **env_kw)
    m = dict(label="behaviour clone of the expert", seed=a.seed, rounds=log, critic_mse=vmse, expert=EXPERT,
             env={**tr.ENV, **ENV}, ppo=dict(tr.DEFAULTS), selection=dict(chosen=0, rate=final["rate"], final_rate=final["rate"]),
             final={k: v for k, v in final.items() if k != "episodes"}, final_episodes=final["episodes"],
             evals=[dict(timesteps=0, rate=final["rate"], finished=final["finished"])],
             wall_s=round(time.perf_counter() - t0, 1))
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    print(f"BC: val landed {final['rate']:.0%} ({final['landed']}/{final['jumps']}), finished {final['finished']:.0%}")


if __name__ == "__main__":
    main()
