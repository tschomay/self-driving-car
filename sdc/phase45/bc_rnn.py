"""Phase 4.5 E4(b): DAgger copying of the expert into the owner's GRU network (sdc/phase4/my_gru.py).

    python -m sdc.phase45.bc_rnn --out runs/p45-e4-b-s0 --seed 0 [--env KEY=JSON ...] [--expert JSON]

Like sdc/phase4/bc.py, but the student has memory. Round 0 drives with the expert, later rounds with the student
(plus a little noise); the expert labels every step. The student's GRU runs alongside in every round, and its memory
at each step is stored, so training cuts each car's run into 64-step chunks (the owner's seq_len) that start from the
memory the student really had there (truncated backpropagation through time). Memory is zeroed at each new course
(the owner's reset_memory). Loss: squared error between tanh(actor mean) and the expert's action. No critic training
(E4 copies only). Writes model.pt and metrics.json loadable with sdc.phase4.train_rnn.load.
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
from sdc.phase4 import my_gru
from sdc.phase4 import train as tr
from sdc.phase4 import train_rnn as trr

ENV = dict(steer_cost=0.03, jump_bonus=10.0, max_steps=1500)
EXPERT = dict(plan=False, v_target=7.4, kd=1.0)


def collect(net, expert, n_steps, seed, use_expert, noise=0.1, n=16):
    """Per step: obs, expert action, episode-started flag, and the student's memory before the step."""
    b = je.JumpBatch(n, seed=seed, **{k: v for k, v in ENV.items() if k != "val_dr"})
    b.reset()
    rng = np.random.default_rng(seed)
    h = net.init_memory(n)
    started = torch.ones(n)
    O, A, S, Hs = [], [], [], []
    for _ in range(n_steps // n):
        o = b.obs().copy()
        a_exp = expert(b)
        with torch.no_grad():
            Hs.append(h.clone())
            a_in, _, h = net.step(tr._t(o), h, started)
            a_stu = np.tanh(net.actor(a_in).numpy())
        S.append(started.numpy().copy())
        a = a_exp if use_expert else a_stu
        a = np.clip(a + noise * rng.standard_normal(a.shape), -1, 1)
        _, _, term, trunc, _ = b.step(a)
        O.append(o)
        A.append(a_exp.astype(np.float32))
        started = torch.as_tensor((term | trunc).astype(np.float32))
    # (T, n, ...) -> per car chunks of seq_len
    return np.stack(O), np.stack(A), np.stack(S), torch.stack(Hs)   # H: (T, n_mem, n, H)


def chunks(O, A, S, H, L):
    T = (len(O) // L) * L
    out = []
    for i in range(O.shape[1]):
        for t in range(0, T, L):
            out.append((O[t:t + L, i], A[t:t + L, i], S[t:t + L, i], H[t, :, i]))
    return out


def fit(net, data, epochs, lr=1e-3, seed=0, batch=32):
    torch.manual_seed(seed)
    params = list(net.cells.parameters()) + list(net.actor.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    O = torch.as_tensor(np.clip(np.stack([d[0] for d in data]), -tr.OBS_CLIP, tr.OBS_CLIP))   # (N, L, D)
    A = torch.as_tensor(np.stack([d[1] for d in data]))
    S = torch.as_tensor(np.stack([d[2] for d in data]))
    H0 = torch.stack([d[3] for d in data], 1)                                                  # (n_mem, N, H)

    def loss_of(idx):
        h = H0[:, idx]
        tot = 0.0
        for t in range(O.shape[1]):
            a_in, _, h = net.step(O[idx, t], h, S[idx, t])
            tot = tot + ((torch.tanh(net.actor(a_in)) - A[idx, t]) ** 2).mean()
        return tot / O.shape[1]

    for _ in range(epochs):
        perm = torch.randperm(len(O))
        for s in range(0, len(O), batch):
            loss = loss_of(perm[s:s + batch])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
    with torch.no_grad():
        return float(np.mean([float(loss_of(torch.arange(s, min(s + 256, len(O))))) for s in range(0, len(O), 256)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--expert", default=None)
    a = ap.parse_args()
    for kv in a.env:
        k, v = kv.split("=", 1)
        ENV[k] = json.loads(v)
    if a.expert:
        EXPERT.clear()
        EXPERT.update(json.loads(a.expert))
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    t0 = time.perf_counter()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not my_gru.ready():
        raise SystemExit("my_gru.py has OWNER TODOs")
    L = my_gru.CONFIG["seq_len"]
    obs_dim = je.JumpBatch(1, **{k: v for k, v in ENV.items() if k != "val_dr"}).obs_dim
    net = trr.RecurrentActorCritic(obs_dim=obs_dim)
    expert = dr.ExpertDriver(**EXPERT)
    env_kw = dict(fan=ENV.get("fan", False), servo=ENV.get("servo", False), real=ENV.get("real", ()),
                  sensors=ENV.get("sensors", "ideal"), dr=ENV.get("val_dr"))
    if ENV.get("attitude"):
        env_kw["attitude"] = True
    data, log = [], []
    for r in range(a.rounds):
        O, A, S, H = collect(net, expert, a.steps, a.seed * 1000 + r, use_expert=(r == 0))
        data += chunks(O, A, S, H, L)
        mse = fit(net, data, a.epochs, seed=a.seed)
        sc = dr.score(dr.run(trr.GreedyRNN(net, env_kw), crs.VAL_SEEDS, n_envs=1))
        log.append(dict(round=r, samples=len(data) * L, mse=round(mse, 4), rate=sc["rate"], finished=sc["finished"],
                        wall_s=round(time.perf_counter() - t0)))
        print(log[-1], flush=True)
    torch.save(net.state_dict(), out / "model.pt")
    final = dr.score(eps := dr.run(trr.GreedyRNN(net, env_kw), crs.VAL_SEEDS, n_envs=1, starts=dr.STARTS))
    m = dict(label="GRU copy of the expert", seed=a.seed, rounds=log, expert=EXPERT, gru=my_gru.CONFIG,
             env={**tr.ENV, **ENV}, ppo=dict(tr.DEFAULTS), final=final, final_episodes=eps,
             selection=dict(chosen=0, rate=final["rate"], final_rate=final["rate"]),
             evals=[dict(timesteps=0, rate=final["rate"], finished=final["finished"])],
             wall_s=round(time.perf_counter() - t0, 1))
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    print(f"GRU copy: val landed {final['rate']:.0%} ({final['landed']}/{final['jumps']}), finished {final['finished']:.0%}")


if __name__ == "__main__":
    main()
