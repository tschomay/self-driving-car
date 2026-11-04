"""Phase 0 training: PPO (Stable-Baselines3) on CartPole-v1, with metrics for the report.

Usage:
    python -m sdc.phase0.train --preset tuned --seed 0 --out runs/e1-tuned-s0

Each run directory gets:
    metrics.json  config, per-episode training returns, periodic held-out evals, timings
    model.zip     the trained policy (not committed; runs/ is gitignored)
    rollout.gif   one held-out episode of the final policy

"Held-out" for CartPole means evaluation episodes are reset with seeds the
training envs never use (training seeds are < 10_000, eval seeds start at 10_000).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

ENV_ID = "CartPole-v1"
SOLVED_RETURN = 475.0
EVAL_SEED_BASE = 10_000


def linear(start: float):
    """Schedule that decays linearly from `start` to 0 over training."""
    return lambda progress_remaining: progress_remaining * start


# Hyperparameter presets. "default" is SB3's out-of-the-box PPO with one env.
# "tuned" follows the RL Zoo's published CartPole-v1 settings.
PRESETS: dict[str, dict] = {
    "default": dict(n_envs=1, ppo={}),
    "tuned": dict(
        n_envs=8,
        ppo=dict(
            n_steps=32,
            batch_size=256,
            gae_lambda=0.8,
            gamma=0.98,
            n_epochs=20,
            ent_coef=0.0,
            learning_rate=("linear", 1e-3),
            clip_range=("linear", 0.2),
        ),
    ),
}


def _resolve(kwargs: dict) -> dict:
    out = {}
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)) and len(v) == 2 and v[0] == "linear":
            out[k] = linear(float(v[1]))
        else:
            out[k] = v
    return out


def evaluate(policy_fn, n_episodes: int, seed_base: int = EVAL_SEED_BASE) -> list[float]:
    """Run `n_episodes` held-out episodes; policy_fn(obs) -> action."""
    env = gym.make(ENV_ID)
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + i)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy_fn(obs))
            total += r
            done = term or trunc
        returns.append(total)
    env.close()
    return returns


def record_gif(policy_fn, path: Path, seed: int = EVAL_SEED_BASE + 999, every: int = 2) -> float:
    """Save one held-out episode as a GIF (every `every`-th frame, downscaled 2x)."""
    import imageio.v2 as imageio

    env = gym.make(ENV_ID, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, done, total, t = [], False, 0.0, 0
    while not done:
        if t % every == 0:
            frames.append(env.render()[::2, ::2])
        obs, r, term, trunc, _ = env.step(policy_fn(obs))
        total += r
        done = term or trunc
        t += 1
    frames.append(env.render()[::2, ::2])
    env.close()
    imageio.mimsave(path, frames, duration=1000 * every / 50, loop=0)  # CartPole runs at 50 Hz
    return total


class MetricsCallback(BaseCallback):
    """Logs every finished training episode and runs periodic held-out evals."""

    def __init__(self, eval_every: int, eval_episodes: int, stop_when_solved: bool):
        super().__init__()
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.stop_when_solved = stop_when_solved
        self.episodes: list[dict] = []
        self.evals: list[dict] = []
        self.solved_at: dict | None = None
        self._next_eval = 0
        self._t0 = time.perf_counter()

    def _policy(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def _eval(self) -> bool:
        rets = evaluate(self._policy, self.eval_episodes)
        row = dict(
            timesteps=int(self.num_timesteps),
            wall_s=round(time.perf_counter() - self._t0, 2),
            mean=float(np.mean(rets)),
            min=float(np.min(rets)),
            max=float(np.max(rets)),
        )
        self.evals.append(row)
        if self.solved_at is None and row["mean"] >= SOLVED_RETURN:
            self.solved_at = row
            return self.stop_when_solved
        return False

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep is not None:
                self.episodes.append(
                    dict(timesteps=int(self.num_timesteps), ret=float(ep["r"]), length=int(ep["l"]))
                )
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_every
            if self._eval():
                return False  # stop training
        return True


def train(
    preset: str = "tuned",
    seed: int = 0,
    total_timesteps: int = 100_000,
    out: str | Path = "runs/smoke",
    overrides: dict | None = None,
    eval_every: int = 5_000,
    eval_episodes: int = 10,
    final_eval_episodes: int = 100,
    stop_when_solved: bool = False,
    gif: bool = True,
    label: str | None = None,
) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(PRESETS[preset]))  # deep copy
    overrides = dict(overrides or {})
    n_envs = int(overrides.pop("n_envs", cfg.pop("n_envs")))
    cfg["ppo"].update(overrides)

    t0 = time.perf_counter()
    venv = make_vec_env(ENV_ID, n_envs=n_envs, seed=seed)
    model = PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0, **_resolve(cfg["ppo"]))
    cb = MetricsCallback(eval_every, eval_episodes, stop_when_solved)
    model.learn(total_timesteps=total_timesteps, callback=cb)
    train_s = time.perf_counter() - t0
    model.save(out / "model.zip")

    def policy(obs):
        return int(model.predict(obs, deterministic=True)[0])

    final = evaluate(policy, final_eval_episodes)
    gif_return = record_gif(policy, out / "rollout.gif") if gif else None

    metrics = dict(
        label=label or out.name,
        env=ENV_ID,
        preset=preset,
        seed=seed,
        n_envs=n_envs,
        ppo={k: v for k, v in cfg["ppo"].items()},
        total_timesteps=total_timesteps,
        trained_timesteps=int(model.num_timesteps),
        train_wall_s=round(train_s, 2),
        episodes=cb.episodes,
        evals=cb.evals,
        solved_at=cb.solved_at,
        final_eval=dict(
            episodes=final_eval_episodes,
            seed_base=EVAL_SEED_BASE,
            mean=float(np.mean(final)),
            std=float(np.std(final)),
            min=float(np.min(final)),
            returns=final,
        ),
        gif_return=gif_return,
        solved_threshold=SOLVED_RETURN,
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def random_baseline(out: str | Path = "runs/random", n_episodes: int = 100, seed: int = 0) -> dict:
    """A uniformly random policy: the floor every learning curve should beat."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rets = evaluate(lambda obs: int(rng.integers(2)), n_episodes)
    gif_return = record_gif(lambda obs: int(rng.integers(2)), out / "rollout.gif")
    metrics = dict(
        label="random policy",
        env=ENV_ID,
        preset="random",
        seed=seed,
        episodes=[],
        evals=[],
        solved_at=None,
        final_eval=dict(
            episodes=n_episodes,
            seed_base=EVAL_SEED_BASE,
            mean=float(np.mean(rets)),
            std=float(np.std(rets)),
            min=float(np.min(rets)),
            returns=rets,
        ),
        gif_return=gif_return,
        solved_threshold=SOLVED_RETURN,
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="tuned", choices=[*PRESETS, "random"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--out", default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a PPO kwarg, e.g. --set learning_rate=3e-3 --set gamma=0.9")
    p.add_argument("--stop-when-solved", action="store_true")
    p.add_argument("--label", default=None)
    a = p.parse_args()
    out = a.out or f"runs/{a.preset}-s{a.seed}"
    if a.preset == "random":
        m = random_baseline(out, seed=a.seed)
    else:
        overrides = {}
        for kv in a.set:
            k, v = kv.split("=", 1)
            overrides[k] = json.loads(v)
        m = train(a.preset, a.seed, a.steps, out, overrides, stop_when_solved=a.stop_when_solved, label=a.label)
    fe = m["final_eval"]
    print(f"{m['label']}: held-out mean return {fe['mean']:.1f} ± {fe['std']:.1f} over {fe['episodes']} eps"
          + (f", trained {m['trained_timesteps']} steps in {m['train_wall_s']:.0f}s" if "train_wall_s" in m else ""))


if __name__ == "__main__":
    main()
