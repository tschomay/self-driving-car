"""Behavior cloning: train the camera policy to copy the Phase 1 ray expert (supervised learning).

Usage:
    python -m sdc.phase2.bc --seed 0 --out runs/p2-e3-clean-s0                 # A: expert's own driving
    python -m sdc.phase2.bc --seed 0 --noise 0.3 --out runs/p2-e3-noisy-s0     # B: wobbly driving, expert labels

Data: `cars` cars on random training tracks (seeds < 10000), `steps` steps each. At every
step we record the camera image, the speed, and the expert's action for that state (the
expert itself reads rays). The car then executes the expert's action plus Gaussian noise
of std `noise` (0 = the expert drives itself). Labels never contain the noise.

Training: the network is SB3's PPO camera policy (NatureCNN + actor MLP), so a cloned
policy can be fine-tuned with PPO afterwards (`python -m sdc.phase2.train --init <dir>/model.zip`).
Loss: mean squared error between the policy's mean action and the expert's action.
The policy's exploration noise (log_std) is set to `init_std` for later PPO fine-tuning.

Output: metrics.json (in the same format as PPO runs, plus per-epoch loss and held-out
completion), model.zip, lap GIFs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sdc.phase2.train import ENV_DEFAULTS, gifs, policy_of
from sdc.racer import evaluate, pursuit, sim

EXPERT = "runs/p1-e1-s1/model.zip"   # Phase 1 SB3 PPO on rays: 100% held-out, 0.94× pure pursuit


def collect(expert, cars: int, steps: int, noise: float, seed: int):
    from sdc.racer import camera

    b = sim.RacerBatch(cars, seed=seed)
    b.reset()
    cam = camera.Camera()
    rng = np.random.default_rng(seed)
    imgs, speeds, acts = [], [], []
    crashes = 0
    for _ in range(steps):
        a = np.clip(expert.predict(b.obs(), deterministic=True)[0], -1, 1)
        imgs.append(cam.batch_views(b))
        speeds.append(b.v / sim.V_MAX)
        acts.append(a)
        _, _, term, _, _ = b.step(np.clip(a + rng.normal(0, noise, a.shape), -1, 1) if noise else a)
        crashes += int(term.sum())
    return (np.concatenate(imgs)[:, None], np.concatenate(speeds)[:, None].astype(np.float32),
            np.concatenate(acts).astype(np.float32), crashes)


def train(seed: int = 0, out: str | Path = "runs/p2-bc", noise: float = 0.0, cars: int = 16, steps: int = 4000,
          epochs: int = 20, lr: float = 3e-4, batch: int = 256, init_std: float = 0.5, label: str | None = None,
          expert_path: str = EXPERT, threads: int = 1, gif: bool = True) -> dict:
    import torch
    from stable_baselines3 import PPO

    from sdc.racer.camera_env import CameraVecEnv

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    expert = PPO.load(expert_path, device="cpu")
    X, V, Y, crashes = collect(expert, cars, steps, noise, seed=1000 + seed)
    collect_s = time.perf_counter() - t0

    env_cfg = dict(ENV_DEFAULTS)
    model = PPO("MultiInputPolicy", CameraVecEnv(2, seed=seed), seed=seed, device="cpu", verbose=0,
                n_steps=256, batch_size=256)
    pol = model.policy
    with torch.no_grad():
        pol.log_std.fill_(float(np.log(init_std)))
    params = [p for n, p in pol.named_parameters() if not n.startswith(("mlp_extractor.value_net", "value_net",
                                                                         "log_std"))]
    opt = torch.optim.Adam(params, lr=lr)

    def mean_action(img, v):
        feats = pol.extract_features({"image": img, "speed": v}, pol.features_extractor)  # scales /255 itself
        return pol.action_net(pol.mlp_extractor.forward_actor(feats))

    Xt, Vt, Yt = torch.from_numpy(X), torch.from_numpy(V), torch.from_numpy(Y)
    n = len(Xt)
    hist = []
    t1 = time.perf_counter()
    for ep in range(epochs):
        perm = torch.randperm(n)
        losses = []
        pol.train()
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            loss = (mean_action(Xt[j], Vt[j]) - Yt[j]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        pol.eval()
        row = dict(epoch=ep + 1, loss=round(float(np.mean(losses)), 5), wall_s=round(time.perf_counter() - t1, 1))
        if (ep + 1) % 5 == 0 or ep + 1 == epochs:
            row["completion"] = evaluate.run_laps(policy_of(model, env_cfg))["completion"]
        hist.append(row)
        print(row, flush=True)
    train_s = time.perf_counter() - t1
    model.save(out / "model.zip")

    baseline = evaluate.run_laps(pursuit.PurePursuit())
    final = evaluate.run_laps(policy_of(model, env_cfg))
    m = dict(label=label or out.name, preset="bc", seed=seed, threads=threads, n_envs=cars,
             ppo=dict(method="behavior cloning", epochs=epochs, lr=lr, batch=batch, init_std=init_std),
             env=dict(noise=noise, expert=expert_path, samples=int(n), cars=cars, steps_per_car=steps),
             total_timesteps=int(n), trained_timesteps=int(n), train_wall_s=round(collect_s + train_s, 1),
             collect_wall_s=round(collect_s, 1), data_crashes=crashes, bc_epochs=hist, episodes=[], evals=[],
             final_eval=final, baseline=baseline, compare=evaluate.compare(final, baseline))
    m["gifs"] = gifs(model, env_cfg, out, f"{m['label']}, seed {seed}", final) if gif else []
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--cars", type=int, default=16)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--init-std", type=float, default=0.5)
    p.add_argument("--label", default=None)
    p.add_argument("--expert", default=EXPERT)
    a = p.parse_args()
    m = train(a.seed, a.out, a.noise, a.cars, a.steps, a.epochs, a.lr, init_std=a.init_std, label=a.label,
              expert_path=a.expert)
    fe = m["final_eval"]
    print(f"{m['label']}: held-out completion {fe['completion']:.0%}, mean lap {fe['mean_lap_time']} s, "
          f"ratio {m['compare']['lap_time_ratio']}; {m['trained_timesteps']} expert samples, "
          f"{m['train_wall_s']:.0f}s; expert left the road {m['data_crashes']} times while recording")


if __name__ == "__main__":
    main()
