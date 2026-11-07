"""Stable-Baselines3 adapter: exposes RacerBatch as an SB3 VecEnv (no per-car Python envs)."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from sdc.racer import sim


class RacerVecEnv(VecEnv):
    def __init__(self, n_envs: int, **sim_kwargs):
        self.batch = sim.RacerBatch(n_envs, **sim_kwargs)
        obs_space = spaces.Box(0.0, 1.0, (sim.OBS_DIM,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        super().__init__(n_envs, obs_space, act_space)
        self._actions = None

    def reset(self):
        return self.batch.reset()

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, r, term, trunc, infos = self.batch.step(self._actions)
        return obs, r, term | trunc, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * len(self._get_indices(indices))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))

    def seed(self, seed=None):
        if seed is not None:
            self.batch.rng = np.random.default_rng(seed)
        return [seed] * self.num_envs
