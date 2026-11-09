"""Stable-Baselines3 adapter: exposes RacerBatch as an SB3 VecEnv (no per-car Python envs)."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from sdc.racer import sim


class RacerVecEnv(VecEnv):
    def __init__(self, n_envs: int, batch_cls=sim.RacerBatch, obs_keep=None, **sim_kwargs):
        """batch_cls: sim.RacerBatch or dynamics.DynamicRacerBatch; obs_keep: observation indices to keep."""
        self.batch = batch_cls(n_envs, **sim_kwargs)
        self.keep = None if obs_keep is None else np.asarray(obs_keep)
        dim = self.batch.obs().shape[1] if self.keep is None else len(self.keep)
        obs_space = spaces.Box(-np.inf, np.inf, (dim,), np.float32) if batch_cls is not sim.RacerBatch \
            else spaces.Box(0.0, 1.0, (dim,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        super().__init__(n_envs, obs_space, act_space)
        self._actions = None

    def _o(self, obs):
        return obs if self.keep is None else obs[:, self.keep]

    def reset(self):
        return self._o(self.batch.reset())

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, r, term, trunc, infos = self.batch.step(self._actions)
        if self.keep is not None:
            for i in np.flatnonzero(term | trunc):
                infos[i]["terminal_observation"] = infos[i]["terminal_observation"][self.keep]
        return self._o(obs), r, term | trunc, infos

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
