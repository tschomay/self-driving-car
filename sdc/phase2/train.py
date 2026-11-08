"""Phase 2 training: Stable-Baselines3 PPO driving from camera images (NatureCNN encoder).

Usage:
    python -m sdc.phase2.train --seed 0 --steps 2000000 --out runs/p2-e1-s0
    python -m sdc.phase2.train --env n_frames=4 --env speed=false ...    # frame stack, no speed number
    python -m sdc.phase2.train --out runs/p2-e1-s0 --resume                # continue from checkpoint
    python -m sdc.phase2.train --init runs/p2-e3-noisy-s0/model.zip ...   # fine-tune a cloned policy

Same tracks, physics, reward and lap evaluation as Phase 1; only the observation changes
(sdc/racer/camera.py). Each run directory gets metrics.json (written at every checkpoint),
ckpt.zip (model + optimizer, for --resume), model.zip and lap GIFs with a camera inset.

Long runs: a checkpoint is saved every --ckpt-every steps. --resume continues from it
(the run is then no longer bit-for-bit reproducible from the seed).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sdc.phase1.train import GIF_TRACK, _resolve, _summary
from sdc.racer import evaluate, pursuit, render

PRESETS: dict[str, dict] = {
    # Phase 1's "base" PPO settings, with SB3's default image network (NatureCNN).
    "base": dict(n_envs=8, ppo=dict(n_steps=256, batch_size=256)),
}
ENV_DEFAULTS = dict(tracks="random", reward="progress", crash_penalty=0.0, max_steps=1000, n_frames=1, speed=True, shift=0)
CAM_KEYS = ("n_frames", "speed")


def policy_of(model, env_cfg):
    from sdc.racer.camera_env import CameraPolicy

    return CameraPolicy(lambda o: model.predict(o, deterministic=True)[0], env_cfg["n_frames"], env_cfg["speed"])


def gifs(model, env_cfg, out: Path, label: str, final: dict) -> list[dict]:
    tracks = [GIF_TRACK]
    failed = [r["track"] for r in final["tracks"] if not r["finished"] and r["track"] != GIF_TRACK]
    if failed:
        tracks.append(failed[0])
    rec = evaluate.run_laps(policy_of(model, env_cfg), tracks, record=True)
    names = []
    for seed, traj, row in zip(tracks, rec["trajectories"], rec["tracks"]):
        name = f"lap-{seed}.gif"
        render.lap_gif(seed, traj, row, label, out / name)
        names.append(dict(file=name, track=seed, finished=row["finished"], crashed=row["crashed"],
                          lap_time=row["lap_time"], laps=row["laps"]))
    return names


def train(preset: str = "base", seed: int = 0, total_timesteps: int = 2_000_000, out: str | Path = "runs/p2",
          overrides: dict | None = None, env: dict | None = None, eval_every: int = 50_000,
          ckpt_every: int = 100_000, gif: bool = True, label: str | None = None, threads: int = 1,
          resume: bool = False, init: str | None = None) -> dict:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    from sdc.racer.camera_env import CameraVecEnv

    torch.set_num_threads(threads)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(PRESETS[preset]))
    overrides = dict(overrides or {})
    n_envs = int(overrides.pop("n_envs", cfg.pop("n_envs")))
    cfg["ppo"].update(overrides)
    env_cfg = {**ENV_DEFAULTS, **(env or {})}
    baseline = evaluate.run_laps(pursuit.PurePursuit())
    m = dict(label=label or out.name, preset=preset, seed=seed, threads=threads, n_envs=n_envs, ppo=cfg["ppo"],
             env=env_cfg, total_timesteps=total_timesteps, episodes=[], evals=[], baseline=baseline,
             train_wall_s=0.0, resumed_at=[])
    venv = CameraVecEnv(n_envs, seed=seed, **env_cfg)
    ckpt = out / "ckpt.zip"
    if resume and ckpt.exists():
        m = json.loads((out / "metrics.json").read_text())
        model = PPO.load(ckpt, env=venv, device="cpu")
        m["resumed_at"].append(int(model.num_timesteps))
    else:
        model = PPO("MultiInputPolicy", venv, seed=seed, device="cpu", verbose=0, **_resolve(cfg["ppo"]))
        if init:   # start from another policy's weights (e.g. behavior cloning); the critic comes along untrained
            model.policy.load_state_dict(PPO.load(init, device="cpu").policy.state_dict())
            m["init"] = init

    def save(final: bool = False):
        m["trained_timesteps"] = int(model.num_timesteps)
        model.save(ckpt)
        (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))

    class Metrics(BaseCallback):
        def __init__(self):
            super().__init__()
            self._t0 = time.perf_counter()
            self._wall0 = m["train_wall_s"]
            self._next_eval = (m["evals"][-1]["timesteps"] + eval_every) if m["evals"] else 0
            self._next_ckpt = model.num_timesteps + ckpt_every

        def wall(self):
            return self._wall0 + time.perf_counter() - self._t0

        def _eval(self):
            m["evals"].append(dict(timesteps=int(self.num_timesteps), wall_s=round(self.wall(), 1),
                                   **_summary(evaluate.run_laps(policy_of(self.model, env_cfg)))))

        def _on_step(self):
            for info in self.locals["infos"]:
                ep = info.get("episode")
                if ep is not None:
                    m["episodes"].append(dict(timesteps=int(self.num_timesteps), ret=round(ep["r"], 3),
                                              length=ep["l"], laps=round(info["laps"], 3),
                                              crashed=info["crashed"], speed=round(info["mean_speed"], 2)))
            if self.num_timesteps >= self._next_eval:
                self._next_eval += eval_every
                self._eval()
            if self.num_timesteps >= self._next_ckpt:
                self._next_ckpt += ckpt_every
                m["train_wall_s"] = round(self.wall(), 1)
                save()
            return True

    cb = Metrics()
    remaining = total_timesteps - model.num_timesteps
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=cb, reset_num_timesteps=False)
    if not m["evals"] or m["evals"][-1]["timesteps"] < model.num_timesteps:
        cb._eval()
    m["train_wall_s"] = round(cb.wall(), 1)
    model.save(out / "model.zip")

    final = evaluate.run_laps(policy_of(model, env_cfg))
    m.update(final_eval=final, compare=evaluate.compare(final, baseline))
    m["gifs"] = gifs(model, env_cfg, out, f"{m['label']}, seed {seed}", final) if gif else []
    save()
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="base", choices=list(PRESETS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--out", default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="PPO kwarg override (JSON value)")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help=f"racer/camera option (JSON value); defaults {ENV_DEFAULTS}")
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--ckpt-every", type=int, default=100_000)
    p.add_argument("--label", default=None)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init", default=None, help="model.zip whose policy weights to start from (e.g. a BC run)")
    a = p.parse_args()

    def parse(kvs):
        return {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in kvs)}

    out = a.out or f"runs/p2-{a.preset}-s{a.seed}"
    m = train(a.preset, a.seed, a.steps, out, parse(a.set), parse(a.env), a.eval_every, a.ckpt_every,
              label=a.label, threads=a.threads, resume=a.resume, init=a.init)
    c, fe = m["compare"], m["final_eval"]
    ratio = f"{c['lap_time_ratio']:.3f}" if c["lap_time_ratio"] else "n/a"
    print(f"{m['label']}: held-out completion {fe['completion']:.0%}, mean lap {fe['mean_lap_time']} s, "
          f"lap-time ratio vs pure pursuit {ratio}; trained {m['trained_timesteps']} steps in "
          f"{m['train_wall_s']:.0f}s")


if __name__ == "__main__":
    main()
