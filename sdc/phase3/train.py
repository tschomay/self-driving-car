"""Phase 3 training: SB3 PPO on the dynamic racer (tire slip, dirt, delay, noise, randomized physics).

Usage:
    python -m sdc.phase3.train --seed 0 --steps 1000000 --out runs/p3-e1-s0
    python -m sdc.phase3.train --env physics='"train"' ...            # domain randomization
    python -m sdc.phase3.train --env physics='{"grip": [0.7, 1.1]}' ...
    python -m sdc.phase3.train --env obs_keep='[0,1,2,3,4,5,6,7,8,9,10,11]' ...   # rays + speed only
    python -m sdc.phase3.train --preset pursuit --out runs/p3-pursuit             # baseline only
    python -m sdc.phase3.train --preset phase1 --out runs/p3-phase1               # Phase 1 policy, unchanged

Each run directory gets metrics.json (same format as Phase 1, plus `gate_laps` and
`sweep`), model.zip, and lap GIFs. Held-out evaluation uses tracks 10000-10019 with
their dirt patches, at nominal physics unless --eval-physics says otherwise.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sdc.phase3 import robust
from sdc.racer import dynamics as dy
from sdc.racer import render
from sdc.racer import track as trk

GIF_TRACK = trk.HELD_OUT_SEEDS[0]
GATE = 0.85
PRESETS: dict[str, dict] = {
    # Phase 1's SB3 settings (8 cars, 256 steps per update), γ = 0.99.
    "base": dict(n_envs=8, ppo=dict(n_steps=256, batch_size=256)),
}
ENV_DEFAULTS = dict(tracks="random", reward="progress", crash_penalty=0.0, max_steps=1000, physics="nominal",
                    dirt=True, obs_keep=None)
PHASE1_EXPERT = "runs/p1-e1-s1/model.zip"


def _summary(ev: dict) -> dict:
    return {k: ev[k] for k in ("completion", "mean_lap_time")}


def sb3_policy(model, keep=None):
    keep = None if keep is None else np.asarray(keep)

    def act(b):
        o = b.obs()
        return model.predict(o if keep is None else o[:, keep], deterministic=True)[0]
    return act


def compare(final: dict, baseline: dict, gate: dict | None) -> dict:
    pairs = [(a["lap_time"], b["lap_time"]) for a, b in zip(final["tracks"], baseline["tracks"])
             if a["finished"] and b["finished"]]
    ratio = float(np.mean([a for a, _ in pairs]) / np.mean([b for _, b in pairs])) if pairs else None
    return dict(completion=final["completion"], lap_time_ratio=ratio,
                gate_completion=gate["completion"] if gate else None,
                gate=bool(gate and gate["completion"] >= GATE))


def gifs(policy, out: Path, label: str, final: dict, physics="nominal", tracks=None) -> list[dict]:
    """Lap GIF on the standard held-out track, plus the first failed held-out track if any."""
    tracks = list(tracks or [GIF_TRACK])
    failed = [r["track"] for r in final["tracks"] if not r["finished"] and r["track"] not in tracks]
    if failed:
        tracks.append(failed[0])
    rec = robust.laps(policy, physics=physics, seeds=tracks, record=True)
    names = []
    for seed, traj, row in zip(tracks, rec["trajectories"], rec["tracks"]):
        name = f"lap-{seed}.gif"
        render.lap_gif(seed, traj, row, label, out / name, dirt=dy.dirt_mask(seed))
        names.append(dict(file=name, track=seed, finished=row["finished"], crashed=row["crashed"],
                          lap_time=row["lap_time"], laps=row["laps"]))
    return names


def finish(policy, out: Path, m: dict, gif: bool = True, sweep: bool = True) -> dict:
    """Final evaluations shared by every Phase 3 run: nominal laps, gate-range laps, sweeps, GIFs."""
    t0 = time.perf_counter()
    m["final_eval"] = robust.laps(policy)
    m["gate_laps"] = robust.gate_laps(policy)
    m["train_range_laps"] = robust.gate_laps(policy, physics=dy.TRAIN_DR, seed=777)
    if sweep:
        m["sweep"] = robust.sweep(policy)
    m["compare"] = compare(m["final_eval"], m["baseline"], m["gate_laps"])
    m["gifs"] = gifs(policy, out, m["label"], m["final_eval"]) if gif else []
    m["eval_wall_s"] = round(time.perf_counter() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    return m


def baseline_laps() -> dict:
    return robust.laps(dy.DynamicPursuit())


def pursuit_run(out="runs/p3-pursuit", gif=True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m = dict(label="hand-coded driver (knows track + grip)", preset="pursuit", seed=0, episodes=[], evals=[],
             baseline=baseline_laps())
    return finish(dy.DynamicPursuit(), out, m, gif)


def phase1_run(out="runs/p3-phase1", expert=PHASE1_EXPERT, gif=True) -> dict:
    """The Phase 1 ray policy (trained with perfect grip), unchanged, driving the dynamic car."""
    from stable_baselines3 import PPO
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = PPO.load(expert, device="cpu")
    pol = dy.phase1_policy(lambda o: model.predict(o, deterministic=True)[0])
    m = dict(label="Phase 1 policy (perfect-grip training), unchanged", preset="phase1", seed=1, episodes=[],
             evals=[], baseline=baseline_laps())
    return finish(pol, out, m, gif)


def train(preset="base", seed=0, total_timesteps=1_000_000, out="runs/p3", overrides=None, env=None,
          eval_every=50_000, gif=True, label=None, threads=1, sweep=True) -> dict:
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
    keep = env_cfg.pop("obs_keep")
    baseline = baseline_laps()

    class Metrics(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episodes, self.evals = [], []
            self._next = 0
            self._t0 = time.perf_counter()

        def _eval(self):
            pol = sb3_policy(self.model, keep)
            self.evals.append(dict(timesteps=int(self.num_timesteps), wall_s=round(time.perf_counter() - self._t0, 2),
                                   **_summary(robust.laps(pol)),
                                   gate_range=robust.gate_laps(pol, draws=1)["completion"]))

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
    venv = RacerVecEnv(n_envs, batch_cls=dy.DynamicRacerBatch, obs_keep=keep, seed=seed, **env_cfg)
    model = PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0, **cfg["ppo"])
    cb = Metrics()
    model.learn(total_timesteps=total_timesteps, callback=cb)
    cb._eval()
    train_s = time.perf_counter() - t0
    model.save(out / "model.zip")
    env_cfg["obs_keep"] = keep
    m = dict(label=label or out.name, preset=preset, seed=seed, threads=threads, n_envs=n_envs, ppo=cfg["ppo"],
             env={k: (dy.physics_range(v).to_dict() if k == "physics" else v) for k, v in env_cfg.items()},
             total_timesteps=total_timesteps, trained_timesteps=int(model.num_timesteps),
             train_wall_s=round(train_s, 2), episodes=cb.episodes, evals=cb.evals, baseline=baseline)
    return finish(sb3_policy(model, keep), out, m, gif, sweep)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="base", choices=[*PRESETS, "pursuit", "phase1"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--out", default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="PPO kwarg override (JSON value)")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help=f"racer option (JSON value); defaults {ENV_DEFAULTS}")
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--label", default=None)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--no-sweep", action="store_true")
    a = p.parse_args()

    def parse(kvs):
        return {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in kvs)}

    out = a.out or f"runs/p3-{a.preset}-s{a.seed}"
    if a.preset == "pursuit":
        m = pursuit_run(out)
    elif a.preset == "phase1":
        m = phase1_run(out)
    else:
        m = train(a.preset, a.seed, a.steps, out, parse(a.set), parse(a.env), a.eval_every, label=a.label,
                  threads=a.threads, sweep=not a.no_sweep)
    c = m["compare"]
    print(f"{m['label']}: nominal held-out {c['completion']:.0%}, gate-range {c['gate_completion']:.0%}, "
          f"lap ratio vs hand-coded {c['lap_time_ratio']}"
          + (f"; trained {m['trained_timesteps']} steps in {m['train_wall_s']:.0f}s" if "train_wall_s" in m else ""))


if __name__ == "__main__":
    main()
