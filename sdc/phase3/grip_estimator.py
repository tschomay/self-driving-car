"""Phase 3 E12: learn to estimate the grip from the last second of driving (Rapid Motor Adaptation, step 2).

E10/E11 trained a policy that is TOLD the grip under the car (an oracle input no real car has). Here a
small network learns to estimate that number from what a real car can sense: the last 1 s (10 steps) of
speed, steering and pedal commands, gyro and accelerometer. It is trained by plain regression against the
sim's true grip. Then the oracle policy, unchanged, drives with the ESTIMATE in place of the true grip.

Data comes from the car being driven by the oracle policy fed with the current estimate (DAgger style):
round 1 uses the true grip, later rounds the estimator's own output, so the estimator also sees the
situations its own mistakes lead to.

    python -m sdc.phase3.grip_estimator --policy runs/p3-e11p-s0 --out runs/p3-e12-s0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sdc.phase3 import train as p3
from sdc.racer import dynamics as dy

HISTORY = 10


class Estimator(nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1) + 1.0   # predicts grip / nominal; centred on 1


def features(obs, b: dy.DynamicRacerBatch):
    """Estimator input: the current own-motion readings plus the remembered ones (no rays, no oracle)."""
    hist_start = b.obs_dim - b.phist.shape[1] * b.phist.shape[2]
    return np.concatenate([obs[:, b.proprio], obs[:, hist_start:]], axis=1).astype(np.float32)


def make_policy(model, est: Estimator | None, b_template: dy.DynamicRacerBatch, use_true=False):
    """The oracle policy driven by the estimate (or the true grip if use_true). Needs a batch built with
    sensors=("accel", "grip") and history=HISTORY; the policy sees [base obs, grip] as it was trained."""
    gc = b_template.grip_col

    def act(b):
        o = b.obs()
        x = o[:, :gc + 1].copy()
        if not use_true:
            with torch.no_grad():
                x[:, gc] = est(torch.as_tensor(features(o, b))).numpy()
        a = model.predict(x, deterministic=True)[0]
        return np.tanh(a)
    return act


def collect(model, est, n_steps, seed, use_true):
    """Drive training-range cars with the policy; return (features, true grip) pairs."""
    b = dy.DynamicRacerBatch(16, physics="train", realism=3, sensors=("accel", "grip"), history=HISTORY, seed=seed)
    b.reset()
    pol = make_policy(model, est, b, use_true)
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_steps // b.n):
        o = b.obs()
        X.append(features(o, b))
        Y.append(o[:, b.grip_col].copy())
        a = pol(b)
        a = np.clip(a + 0.1 * rng.standard_normal(a.shape), -1, 1)   # a little noise for variety
        b.step(a)
    return np.concatenate(X), np.concatenate(Y)


def fit(est, X, Y, epochs=30, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(est.parameters(), lr=lr)
    X, Y = torch.as_tensor(X), torch.as_tensor(Y)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, 512):
            i = perm[s:s + 512]
            loss = ((est(X[i]) - Y[i]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return float(((est(X) - Y) ** 2).mean().sqrt())


def main():
    from stable_baselines3 import PPO
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="an oracle-grip run directory (E11 p)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--steps", type=int, default=60_000, help="samples collected per round")
    p.add_argument("--label", default="oracle policy + learned grip estimate")
    a = p.parse_args()
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = json.loads((Path(a.policy) / "metrics.json").read_text())
    model = PPO.load(Path(a.policy) / "model.zip", device="cpu")
    probe = dy.DynamicRacerBatch(1, realism=3, sensors=("accel", "grip"), history=HISTORY)
    est = Estimator(features(probe.reset(), probe).shape[1])
    Xs, Ys, rounds = [], [], []
    for r in range(a.rounds):
        X, Y = collect(model, est, a.steps, a.seed * 100 + r, use_true=(r == 0))
        Xs.append(X)
        Ys.append(Y)
        rmse = fit(est, np.concatenate(Xs), np.concatenate(Ys), seed=a.seed)
        rounds.append(dict(round=r, samples=int(sum(len(x) for x in Xs)), rmse=round(rmse, 4)))
        print(rounds[-1], flush=True)
    torch.save(est.state_dict(), out / "estimator.pt")
    # check the estimate on fresh driving data (training-range physics, a new seed)
    Xv, Yv = collect(model, est, 20_000, 99_999 + a.seed, use_true=False)
    with torch.no_grad():
        pred = est(torch.as_tensor(Xv)).numpy()
    err = dict(rmse=round(float(np.sqrt(np.mean((pred - Yv) ** 2))), 4),
               mean_abs=round(float(np.mean(np.abs(pred - Yv))), 4),
               corr=round(float(np.corrcoef(pred, Yv)[0, 1]), 3))
    car = dict(realism=3, sensors=("accel", "grip"), history=HISTORY)
    pol = make_policy(model, est, probe)
    m = dict(label=a.label, preset="estimator", seed=a.seed, policy=str(a.policy), rounds=rounds, estimate_error=err,
             env=dict(src["env"], sensors=["accel", "grip"], history=HISTORY), episodes=[], evals=[],
             trained_timesteps=src["trained_timesteps"], train_wall_s=round(time.perf_counter() - t0, 1),
             baseline=p3.baseline_laps(**car))
    m = p3.finish(pol, out, m)
    c = m["compare"]
    print(f"{a.label}: estimate RMSE {err['rmse']}, nominal held-out {c['completion']:.0%}, "
          f"gate-range {c['gate_completion']:.0%}")


if __name__ == "__main__":
    main()
