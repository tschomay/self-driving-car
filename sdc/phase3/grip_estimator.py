"""Phase 3 E13: learn to estimate the grip from driving (Rapid Motor Adaptation, step 2).

E10-E12 trained a policy that is TOLD the grip under the car (an oracle input no real car has). Here a
small network learns to estimate that number from what a real car can sense: speed, steering and pedal
commands, gyro and accelerometer. It is trained by plain regression against the sim's true grip. Then the
oracle policy, unchanged, drives with the ESTIMATE in place of the true grip.

Two kinds of estimator:
  window  sees the last 1 s (10 steps) of readings: can react to a dirt patch, but forgets "today's grip"
  drive   a recurrent network (GRU) that reads the whole drive step by step, from the standing start on: can
          build up an estimate of the day's grip over the out-lap and keep it, while still reacting to dirt

Data comes from the car being driven by the oracle policy fed with the current estimate (DAgger style):
round 1 uses the true grip, later rounds the estimator's own output, so the estimator also sees the
situations its own mistakes lead to.

    python -m sdc.phase3.grip_estimator --policy runs/p3-e12p-s0 --kind drive --out runs/p3-e13-drive-s0
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


class WindowEstimator(nn.Module):
    kind = "window"

    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1) + 1.0   # predicts grip / nominal, centred on 1


class DriveEstimator(nn.Module):
    kind = "drive"

    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.inp = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, h=None):
        """x: (B, T, n_in) -> estimates (B, T), new hidden state."""
        y, h = self.gru(torch.tanh(self.inp(x)), h)
        return self.out(y).squeeze(-1) + 1.0, h


def car_options(kind):
    return dict(realism=3, sensors=("accel", "grip"), history=HISTORY if kind == "window" else 1)


def features(obs, b: dy.DynamicRacerBatch):
    """Estimator input: own-motion readings (current, plus remembered ones if the car has history). No rays,
    no oracle."""
    hist_start = b.obs_dim - b.phist.shape[1] * b.phist.shape[2]
    return np.concatenate([obs[:, b.proprio], obs[:, hist_start:]], axis=1).astype(np.float32)


def make_policy(model, est, use_true=False):
    """The oracle policy driven by the estimate (or by the true grip if use_true). The policy sees
    [base obs, grip] as it was trained. A drive estimator keeps a hidden state per car, reset whenever that
    car starts a new episode (step 0)."""
    state = {}

    def act(b):
        o = b.obs()
        gc = b.grip_col
        x = o[:, :gc + 1].copy()
        f = torch.as_tensor(features(o, b))
        with torch.no_grad():
            if est.kind == "window":
                g = est(f).numpy()
            else:
                h = state.get("h")
                if h is None or h.shape[1] != b.n:
                    h = torch.zeros(1, b.n, est.gru.hidden_size)
                fresh = torch.as_tensor(b.steps == 0)
                h[:, fresh] = 0.0
                y, h = est(f[:, None], h)
                state["h"] = h
                g = y[:, 0].numpy()
        act.estimate = g
        if not use_true:
            x[:, gc] = g
        return np.tanh(model.predict(x, deterministic=True)[0])
    act.estimate = None
    return act


def collect(model, est, n_steps, seed, use_true):
    """Drive training-range cars with the policy. Returns a list of per-episode (features, true grip) arrays."""
    b = dy.DynamicRacerBatch(16, physics="train", seed=seed, **car_options(est.kind))
    b.reset()
    pol = make_policy(model, est, use_true)
    rng = np.random.default_rng(seed)
    cur_x = [[] for _ in range(b.n)]
    cur_y = [[] for _ in range(b.n)]
    seqs = []
    for _ in range(n_steps // b.n):
        o = b.obs()
        f = features(o, b)
        for i in range(b.n):
            cur_x[i].append(f[i])
            cur_y[i].append(o[i, b.grip_col])
        a = pol(b)
        a = np.clip(a + 0.1 * rng.standard_normal(a.shape), -1, 1)   # a little noise for variety
        _, _, term, trunc, _ = b.step(a)
        for i in np.flatnonzero(term | trunc):
            seqs.append((np.array(cur_x[i]), np.array(cur_y[i], np.float32)))
            cur_x[i], cur_y[i] = [], []
    seqs += [(np.array(x), np.array(y, np.float32)) for x, y in zip(cur_x, cur_y) if x]
    return seqs


def fit(est, seqs, epochs=30, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(est.parameters(), lr=lr)
    if est.kind == "window":
        X = torch.as_tensor(np.concatenate([s[0] for s in seqs]))
        Y = torch.as_tensor(np.concatenate([s[1] for s in seqs]))
        for _ in range(epochs):
            perm = torch.randperm(len(X))
            for s in range(0, len(X), 512):
                i = perm[s:s + 512]
                loss = ((est(X[i]) - Y[i]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            return float(((est(X) - Y) ** 2).mean().sqrt())
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(seqs))
        for s in range(0, len(order), 16):
            xb, yb, mb = _pad([seqs[j] for j in order[s:s + 16]])
            pred, _ = est(xb)
            loss = (((pred - yb) ** 2) * mb).sum() / mb.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(est.parameters(), 1.0)
            opt.step()
    se, n = 0.0, 0.0
    with torch.no_grad():
        for s in range(0, len(seqs), 32):
            xb, yb, mb = _pad(seqs[s:s + 32])
            pred, _ = est(xb)
            se += float((((pred - yb) ** 2) * mb).sum())
            n += float(mb.sum())
    return float(np.sqrt(se / n))


def _pad(batch):
    T = max(len(x) for x, _ in batch)
    d = batch[0][0].shape[1]
    xb = np.zeros((len(batch), T, d), np.float32)
    yb = np.zeros((len(batch), T), np.float32)
    mb = np.zeros((len(batch), T), np.float32)
    for k, (x, y) in enumerate(batch):
        xb[k, :len(x)], yb[k, :len(y)], mb[k, :len(y)] = x, y, 1.0
    return torch.as_tensor(xb), torch.as_tensor(yb), torch.as_tensor(mb)


def load(run_dir):
    """(policy, metrics) for a saved estimator run."""
    from stable_baselines3 import PPO
    m = json.loads((Path(run_dir) / "metrics.json").read_text())
    model = PPO.load(Path(m["policy"]) / "model.zip", device="cpu")
    est = _new(m["kind"])
    est.load_state_dict(torch.load(Path(run_dir) / "estimator.pt"))
    return make_policy(model, est), m


def _new(kind):
    probe = dy.DynamicRacerBatch(1, **car_options(kind))
    n_in = features(probe.reset(), probe).shape[1]
    return WindowEstimator(n_in) if kind == "window" else DriveEstimator(n_in)


def main():
    from stable_baselines3 import PPO
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="an oracle-grip run directory (its model.zip is used)")
    p.add_argument("--kind", choices=["window", "drive"], default="drive")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--steps", type=int, default=60_000, help="samples collected per round")
    p.add_argument("--label", default=None)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    t0 = time.perf_counter()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = json.loads((Path(a.policy) / "metrics.json").read_text())
    model = PPO.load(Path(a.policy) / "model.zip", device="cpu")
    est = _new(a.kind)
    data, rounds = [], []
    for r in range(a.rounds):
        data += collect(model, est, a.steps, a.seed * 100 + r, use_true=(r == 0))
        rmse = fit(est, data, seed=a.seed)
        rounds.append(dict(round=r, samples=int(sum(len(s[1]) for s in data)), rmse=round(rmse, 4)))
        print(rounds[-1], flush=True)
    torch.save(est.state_dict(), out / "estimator.pt")
    # check the estimate on fresh driving (training-range physics, a new seed), by time since the start
    check = collect(model, est, 20_000, 99_999 + a.seed, use_true=False)
    with torch.no_grad():
        errs = []
        for x, y in check:
            pred = est(torch.as_tensor(x)).numpy() if est.kind == "window" else est(torch.as_tensor(x)[None])[0][0].numpy()
            errs.append(np.abs(pred - y))
    by_time = {}
    for lo, hi in ((0, 50), (50, 150), (150, 300), (300, 1000)):
        e = np.concatenate([q[lo:hi] for q in errs if len(q) > lo])
        by_time[f"{lo / 10:g}-{hi / 10:g} s"] = round(float(e.mean()), 4) if len(e) else None
    err = dict(mean_abs=round(float(np.concatenate(errs).mean()), 4), mean_abs_by_time=by_time)
    print(err, flush=True)
    label = a.label or f"oracle policy + {a.kind} grip estimate"
    car = car_options(a.kind)
    m = dict(label=label, preset="estimator", kind=a.kind, seed=a.seed, policy=str(a.policy), rounds=rounds,
             estimate_error=err, env=dict(src["env"], sensors=list(car["sensors"]), history=car["history"]),
             episodes=[], evals=[], trained_timesteps=src["trained_timesteps"],
             train_wall_s=round(time.perf_counter() - t0, 1), baseline=p3.baseline_laps(**car))
    pol = make_policy(model, est)
    m = p3.finish(pol, out, m)
    # the same, with an out-lap before the counted lap (the gate protocol from E13 on)
    from sdc.phase3 import robust
    m["outlap"] = dict(final_eval=robust.laps(pol, outlap=True, **car),
                       gate_laps=robust.gate_laps(pol, outlap=True, **car),
                       train_range_laps=robust.gate_laps(pol, physics=dy.TRAIN_DR, seed=777, outlap=True, **car))
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    c = m["compare"]
    print(f"{label}: nominal held-out {c['completion']:.0%}, gate-range {c['gate_completion']:.0%}; with out-lap: "
          f"{m['outlap']['final_eval']['completion']:.0%}, {m['outlap']['gate_laps']['completion']:.0%}")


if __name__ == "__main__":
    main()
