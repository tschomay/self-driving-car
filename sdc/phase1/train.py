"""Phase 1 training: Stable-Baselines3 PPO on the 2D racer, with lap metrics for the report.

Usage:
    python -m sdc.phase1.train --seed 0 --steps 300000 --out runs/p1-e1-s0
    python -m sdc.phase1.train --env reward='"survival"' --env tracks='[0]' --set gamma=0.9 ...
    python -m sdc.phase1.train --preset pursuit --out runs/p1-pursuit   # baseline only, no training

Each run directory gets:
    metrics.json   config, per-episode training stats, periodic held-out lap evals, final eval
    model.zip      the trained policy (not committed; runs/ is gitignored)
    lap-<track>.gif  one lap attempt on a held-out track (and on the first failed one, if any)

Held-out tracks are seeds 10000-10019 (sdc/racer/track.py); training tracks use seeds < 10000.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sdc.racer import evaluate, pursuit, render, sim
from sdc.racer import track as trk

GIF_TRACK = trk.HELD_OUT_SEEDS[0]

PRESETS: dict[str, dict] = {
    # SB3 PPO defaults, except 8 parallel cars and a shorter rollout per update.
    "base": dict(n_envs=8, ppo=dict(n_steps=256, batch_size=256)),
}
ENV_DEFAULTS = dict(tracks="random", reward="progress", crash_penalty=0.0, max_steps=1000)


def linear(start: float):
    return lambda progress_remaining: progress_remaining * start


def _resolve(kwargs: dict) -> dict:
    out = {}
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)) and len(v) == 2 and v[0] == "linear":
            out[k] = linear(float(v[1]))
        else:
            out[k] = v
    return out


def _summary(ev: dict) -> dict:
    return {k: ev[k] for k in ("completion", "mean_lap_time")}


def _gifs(policy, out: Path, label: str, final: dict) -> list[str]:
    """Lap GIF on the standard held-out track, plus the first failed held-out track if any."""
    tracks = [GIF_TRACK]
    failed = [r["track"] for r in final["tracks"] if not r["finished"] and r["track"] != GIF_TRACK]
    if failed:
        tracks.append(failed[0])
    rec = evaluate.run_laps(policy, tracks, record=True)
    names = []
    for seed, traj, row in zip(tracks, rec["trajectories"], rec["tracks"]):
        name = f"lap-{seed}.gif"
        render.lap_gif(seed, traj, row, label, out / name)
        names.append(dict(file=name, track=seed, finished=row["finished"], crashed=row["crashed"],
                          lap_time=row["lap_time"], laps=row["laps"]))
    return names


def pursuit_baseline(out: str | Path = "runs/p1-pursuit", gif: bool = True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pp = pursuit.PurePursuit()
    final = evaluate.run_laps(pp)
    m = dict(label="pure pursuit (hand-coded)", preset="pursuit", seed=0, episodes=[], evals=[], final_eval=final,
             baseline=final, compare=evaluate.compare(final, final))
    m["gifs"] = _gifs(pp, out, m["label"], final) if gif else []
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    return m


def train(preset: str = "base", seed: int = 0, total_timesteps: int = 300_000, out: str | Path = "runs/p1",
          overrides: dict | None = None, env: dict | None = None, eval_every: int = 25_000, gif: bool = True,
          label: str | None = None, threads: int = 1) -> dict:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    from sdc.racer.vec_env import RacerVecEnv

    torch.set_num_threads(threads)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(PRESETS[preset]))
    overrides = dict(overrides or {})
    n_envs = int(overrides.pop("n_envs", cfg.pop("n_envs")))
    cfg["ppo"].update(overrides)
    env_cfg = {**ENV_DEFAULTS, **(env or {})}
    baseline = evaluate.run_laps(pursuit.PurePursuit())
    train_tracks = env_cfg["tracks"] if env_cfg["tracks"] != "random" else None

    def policy_of(model):
        return lambda b: model.predict(b.obs(), deterministic=True)[0]

    class Metrics(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episodes, self.evals = [], []
            self._next = 0
            self._t0 = time.perf_counter()

        def _eval(self):
            row = dict(timesteps=int(self.num_timesteps), wall_s=round(time.perf_counter() - self._t0, 2),
                       **_summary(evaluate.run_laps(policy_of(self.model))))
            if train_tracks:
                row["train_tracks"] = _summary(evaluate.run_laps(policy_of(self.model), train_tracks))
            self.evals.append(row)

        def _on_step(self):
            for info in self.locals["infos"]:
                ep = info.get("episode")
                if ep is not None:
                    self.episodes.append(dict(timesteps=int(self.num_timesteps), ret=round(ep["r"], 3),
                                              length=ep["l"], laps=round(info["laps"], 3),
                                              crashed=info["crashed"], speed=round(info["mean_speed"], 2)))
            if self.num_timesteps >= self._next:
                self._next += eval_every
                self._eval()
            return True

    t0 = time.perf_counter()
    venv = RacerVecEnv(n_envs, seed=seed, **env_cfg)
    model = PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0, **_resolve(cfg["ppo"]))
    cb = Metrics()
    model.learn(total_timesteps=total_timesteps, callback=cb)
    cb._eval()
    train_s = time.perf_counter() - t0
    model.save(out / "model.zip")

    final = evaluate.run_laps(policy_of(model))
    m = dict(label=label or out.name, preset=preset, seed=seed, threads=threads, n_envs=n_envs,
             ppo=cfg["ppo"], env=env_cfg, total_timesteps=total_timesteps,
             trained_timesteps=int(model.num_timesteps), train_wall_s=round(train_s, 2),
             episodes=cb.episodes, evals=cb.evals, final_eval=final, baseline=baseline,
             compare=evaluate.compare(final, baseline))
    if train_tracks:
        m["train_tracks_eval"] = evaluate.run_laps(policy_of(model), train_tracks)
    m["gifs"] = _gifs(policy_of(model), out, f"{m['label']}, seed {seed}", final) if gif else []
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="base", choices=[*PRESETS, "pursuit"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--out", default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="PPO kwarg override (JSON value)")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help=f"racer option (JSON value); defaults {ENV_DEFAULTS}")
    p.add_argument("--eval-every", type=int, default=25_000)
    p.add_argument("--label", default=None)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()

    def parse(kvs):
        d = {}
        for kv in kvs:
            k, v = kv.split("=", 1)
            d[k] = json.loads(v)
        return d

    out = a.out or f"runs/p1-{a.preset}-s{a.seed}"
    if a.preset == "pursuit":
        m = pursuit_baseline(out)
    else:
        m = train(a.preset, a.seed, a.steps, out, parse(a.set), parse(a.env), a.eval_every, label=a.label,
                  threads=a.threads)
    c, fe = m["compare"], m["final_eval"]
    ratio = f"{c['lap_time_ratio']:.3f}" if c["lap_time_ratio"] else "n/a"
    print(f"{m['label']}: held-out completion {fe['completion']:.0%}, mean lap {fe['mean_lap_time']} s, "
          f"lap-time ratio vs pure pursuit {ratio}, gate {'PASS' if c['gate'] else 'fail'}"
          + (f"; trained {m['trained_timesteps']} steps in {m['train_wall_s']:.0f}s" if "train_wall_s" in m else ""))


if __name__ == "__main__":
    main()
