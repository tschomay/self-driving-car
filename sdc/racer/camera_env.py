"""Stable-Baselines3 adapter for the camera car: RacerBatch physics, camera-image observations.

Observation (a dict, for SB3's MultiInputPolicy):
  image  (n_frames, 64, 64) uint8, the last n_frames camera images, newest last
  speed  (1,) float32, speed / V_MAX   (only if speed=True)
With shift > 0, every training image is slid by a random ±shift pixels (DrQ-style augmentation).

Everything else (tracks, physics, reward, episode ends) is the Phase 1 racer.
`CameraPolicy` wraps a trained model as a policy(batch) for evaluate.run_laps.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from sdc.racer import camera, sim


def obs_space(n_frames: int, speed: bool) -> spaces.Dict:
    d = {"image": spaces.Box(0, 255, (n_frames, camera.SIZE, camera.SIZE), np.uint8)}
    if speed:
        d["speed"] = spaces.Box(0.0, 1.0, (1,), np.float32)
    return spaces.Dict(d)


class CameraVecEnv(VecEnv):
    def __init__(self, n_envs: int, n_frames: int = 1, speed: bool = True, shift: int = 0, **sim_kwargs):
        self.batch = sim.RacerBatch(n_envs, **sim_kwargs)
        self.shift = shift   # training-time random shift (DrQ), pixels; lap evaluation never shifts
        self._shift_rng = np.random.default_rng(sim_kwargs.get("seed", 0) + 7)
        self.cam = camera.Camera()
        self.n_frames, self.use_speed = n_frames, speed
        self.stack = camera.FrameStack(n_envs, n_frames)
        super().__init__(n_envs, obs_space(n_frames, speed), spaces.Box(-1.0, 1.0, (2,), np.float32))
        self._actions = None

    def _obs(self, frames, v):
        o = {"image": frames.copy()}
        if self.use_speed:
            o["speed"] = (np.asarray(v, float) / sim.V_MAX).reshape(-1, 1).astype(np.float32)
        return o

    def _views(self, b):
        return camera.random_shift(self.cam.batch_views(b), self.shift, self._shift_rng)

    def reset(self):
        self.batch.reset()
        imgs = self._views(self.batch)
        for i in range(self.num_envs):
            self.stack.reset(i, imgs[i])
        return self._obs(self.stack.frames, self.batch.v)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        _, r, term, trunc, infos = self.batch.step(self._actions)
        done = term | trunc
        b = self.batch
        imgs = self._views(b)
        if done.any():
            # finished cars were already reset onto a new episode: render their last view for SB3's
            # bootstrapping, then restart their frame stack from the new episode's first image
            last = self.stack.frames.copy()
            for i in np.flatnonzero(done):
                fs = infos[i]["final_state"]
                term_img = camera.random_shift(self.cam.view(infos[i]["track"], fs["pos"], fs["heading"]),
                                               self.shift, self._shift_rng)
                last[i] = np.concatenate([last[i, 1:], term_img[None]])
                infos[i]["terminal_observation"] = {k: v[0] for k, v in
                                                    self._obs(last[i:i + 1], [fs["v"]]).items()}
        frames = self.stack.push(imgs)
        for i in np.flatnonzero(done):
            self.stack.reset(i, imgs[i])
        return self._obs(self.stack.frames, b.v), r, done, infos

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


class CameraPolicy:
    """policy(batch) -> actions for evaluate.run_laps, keeping a frame stack per car.

    A car's stack restarts whenever its step counter is 0 (the start of an attempt).
    `last_images` holds the newest image per car, for GIF insets.
    """

    def __init__(self, act, n_frames: int = 1, speed: bool = True, cam: camera.Camera | None = None):
        self.act, self.n_frames, self.use_speed = act, n_frames, speed
        self.cam = cam or camera.Camera()
        self.stack = None
        self.last_images = None

    def __call__(self, b):
        imgs = self.cam.batch_views(b)
        if self.stack is None or self.stack.frames.shape[0] != b.n:
            self.stack = camera.FrameStack(b.n, self.n_frames)
            for i in range(b.n):
                self.stack.reset(i, imgs[i])
        self.stack.push(imgs)
        for i in np.flatnonzero(b.steps == 0):
            self.stack.reset(i, imgs[i])
        self.last_images = imgs
        o = {"image": self.stack.frames.copy()}
        if self.use_speed:
            o["speed"] = (b.v / sim.V_MAX).reshape(-1, 1).astype(np.float32)
        return self.act(o)
