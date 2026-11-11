"""Phase 3.5 E4: behaviour cloning of the hand-coded driver into the owner's network, as a PPO starting point.

    python -m sdc.phase35.bc --out runs/p35-e4-bc-s0 --seed 0

The expert is dy.DynamicPursuit with full knowledge (true grip, dirt map, mass): 25.4 s at nominal. The student
is owner.ActorCritic with E3's inputs (accelerometer, true grip, gripmap), which carry the same grip knowledge.
Training is supervised (no reward):
  - actor: DAgger. Round 0 drives with the expert; later rounds drive with the student (plus a little noise) and
    the expert labels each state. Loss: squared error between tanh(actor mean) and the expert's action.
  - critic: regressed on the discounted return (γ 0.997, the PPO reward with lap bonus 100) of the student's own
    driving, so PPO's first updates see sensible advantages instead of a random value function.
Writes model.pt (a state_dict for my_ppo --set init=...), metrics.json, and the student's validation scores.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sdc.phase1 import my_ppo as owner
from sdc.phase3 import robust
from sdc.racer import dynamics as dy

SENSORS = ("accel", "grip", "gripmap")
CAR = dict(realism=3, sensors=SENSORS, history=1)
ENV = dict(tracks="random", reward="progress", max_steps=1000, physics="train", dirt=True, lap_bonus=100.0)
GAMMA = 0.997


def batch(seed, n=16):
    return dy.DynamicRacerBatch(n, seed=seed, **ENV, **CAR)


def student_policy(net):
    def act(b):
        with torch.no_grad():
            return np.tanh(net.actor(torch.as_tensor(b.obs())).numpy())
    return act


def collect(net, expert, n_steps, seed, use_expert, noise=0.1):
    """States visited by the driver (expert or student) and the expert's action there; also rewards and dones
    (for the critic)."""
    b = batch(seed)
    b.reset()
    rng = np.random.default_rng(seed)
    stud = student_policy(net)
    O, A, R, D = [], [], [], []
    for _ in range(n_steps // b.n):
        o = b.obs()
        a_exp = expert(b)
        a = a_exp if use_expert else stud(b)
        a = np.clip(a + noise * rng.standard_normal(a.shape), -1, 1)
        _, r, term, trunc, _ = b.step(a)
        O.append(o)
        A.append(a_exp.astype(np.float32))
        R.append(r.astype(np.float32))
        D.append((term | trunc))
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
    X, Y = torch.as_tensor(X), torch.as_tensor(Y)
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
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=50_000)
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    t0 = time.perf_counter()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    expert = dy.DynamicPursuit()
    net = owner.ActorCritic(obs_dim=batch(0, 1).obs_dim)
    Xs, Ys, log = [], [], []
    for r in range(a.rounds):
        O, A, _, _ = collect(net, expert, a.steps, a.seed * 1000 + r, use_expert=(r == 0))
        Xs.append(O.reshape(-1, O.shape[-1]))
        Ys.append(A.reshape(-1, 2))
        mse = fit(net.actor, np.concatenate(Xs), np.concatenate(Ys), epochs=15, squash=True, seed=a.seed)
        ev = robust.laps(student_policy(net), **CAR)
        log.append(dict(round=r, samples=int(sum(len(x) for x in Xs)), mse=round(mse, 4),
                        completion=ev["completion"], mean_lap_time=ev["mean_lap_time"]))
        print(log[-1], flush=True)
    # critic: returns of the student's own (noisy, as in PPO) driving
    O, _, R, D = collect(net, expert, 2 * a.steps, a.seed * 1000 + 99, use_expert=False, noise=0.2)
    G = returns(R, D)
    vmse = fit(net.critic, O.reshape(-1, O.shape[-1]), G.reshape(-1), epochs=20, seed=a.seed)
    print(dict(critic_mse=round(vmse, 2), mean_return=round(float(G.mean()), 1)), flush=True)
    torch.save(net.state_dict(), out / "model.pt")
    pol = student_policy(net)
    m = dict(label="behaviour clone of the hand-coded driver", seed=a.seed, rounds=log, critic_mse=vmse,
             env={**ENV, "realism": 3, "sensors": list(SENSORS), "history": 1, "squash": True},
             final_eval=robust.laps(pol, **CAR), gate_laps=robust.gate_laps(pol, **CAR),
             wall_s=round(time.perf_counter() - t0, 1))
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    print(f"BC: nominal {m['final_eval']['completion']:.0%}, {m['final_eval']['mean_lap_time']:.1f} s; "
          f"gate range {m['gate_laps']['completion']:.0%}")


if __name__ == "__main__":
    main()
