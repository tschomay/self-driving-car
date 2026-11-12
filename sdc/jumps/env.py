"""A batch of 3D RC cars on jump courses (one MuJoCo model + data per car), with jump scoring.

API like the 2D racers: reset() -> obs (B, D); step(actions (B, 2) in [-1, 1]) ->
obs, reward, terminated, truncated, infos. Finished episodes reset automatically (a new course);
infos[i] then carries "terminal_observation" and "episode" (the summary, incl. per-jump results).

Actions: [steer, throttle]; steer +1 = full left (0.4 rad), throttle +1 = full drive, -1 = full brake/reverse.

Observation (real sensors only, OBS_DIM = 20):
  accel / 9.81 (3, body frame), gyro / 5 (3), wheel speed front/rear / 10 m/s (2),
  8 forward rays (3° up to 50° down) + left/right rays, distance / 6 m (10, 1.0 = nothing within 6 m),
  last action (2).

A jump (the gate's definition, owner 2026-11-12): all four wheels off the ground for >= 0.3 s.
It is credited to the course jump whose stretch contains the takeoff point (only the first jump
from each ramp counts). It is LANDED if, within 1 s of touchdown, all four wheels are on the
ground together at some moment, the chassis never touches the ground, and at the end of that
second the car is upright (up-axis z >= 0.7) and still driving (forward speed >= 0.5 m/s).
A ramp the car didn't jump (too slow, or it crashed before reaching it) counts as not landed.

Episode ends: finish line (terminated, success), rollover (up-axis z < 0.2), off the lane,
stuck (< 0.3 m/s for 2 s), or the step cap (truncated).

Reward: forward progress in metres + jump_bonus for each landed jump.
"""

from __future__ import annotations

import numpy as np

import mujoco

from sdc.jumps import car as cr
from sdc.jumps import course as crs

OBS_DIM = 20
MIN_AIR = 0.3
LAND_WINDOW = 1.0


def _pitch_roll(R):
    pitch = np.arctan2(R[2, 0], np.hypot(R[0, 0], R[1, 0]))   # nose up > 0
    roll = np.arctan2(R[2, 1], R[2, 2])
    return pitch, roll


class JumpCar:
    """One car on one course."""

    def __init__(self, seed_rng, courses="random", size=1.0, kinds=crs.KINDS):
        self.m = cr.build_model()
        self.d = mujoco.MjData(self.m)
        self.rng = seed_rng
        self.courses = courses
        self.size, self.kinds = size, kinds
        self._course_i = 0
        g = self.m.geom
        self.ground = g("ground").id
        self.tires = np.array([g(f"{w}_tire").id for w in cr.WHEELS])
        self.chassis = g("chassis").id
        self.spin_dof = np.array([self.m.jnt_dofadr[self.m.joint(f"{w}_spin").id] for w in cr.WHEELS])
        s = {self.m.sensor(i).name: (self.m.sensor_adr[i], self.m.sensor_dim[i]) for i in range(self.m.nsensor)}
        self.s_accel, self.s_gyro = s["accel"][0], s["gyro"][0]
        self.ray_sites = np.array([self.m.site(n).id for n in cr.RAY_SITES])

    # ---- episode ---------------------------------------------------------------------
    def reset(self, course_seed=None, x0=0.0):
        if course_seed is None:
            if self.courses == "random":
                course_seed = int(self.rng.integers(0, 2**31 - 1))
            else:
                course_seed = self.courses[self._course_i % len(self.courses)]
                self._course_i += 1
        self.course = crs.make(course_seed, kinds=self.kinds, size=self.size)
        self.m.hfield_data[:] = self.course.hfield().ravel()
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0] = crs.START_X + x0      # x0: start offset (evaluation repeats a course from a few starts)
        self.d.qpos[2] = self.course.height(crs.START_X + x0) + 0.17
        mujoco.mj_forward(self.m, self.d)
        mujoco.mj_step(self.m, self.d, int(0.2 / cr.TIMESTEP))     # settle on the suspension (0.2 s)
        self.t = 0
        self.last_act = np.zeros(2, np.float32)
        self.ret = 0.0
        self.x_prev = float(self.d.qpos[0])
        self.slow = 0
        self.airborne = False
        self.takeoff = None          # (t, x)
        self.pending = None          # landing under evaluation
        self.results = [None] * len(self.course.jumps)   # per course jump: dict or None
        self.air_log = []            # every airborne spell >= MIN_AIR
        self.max_air = 0.0
        return self.obs()

    def contacts(self):
        c = self.d.contact
        n = self.d.ncon
        if n == 0:
            return np.zeros(4, bool), False
        g1, g2 = c.geom1[:n], c.geom2[:n]
        # tires carry a 1 cm detection margin (no force until they touch): a wheel within 1 cm of the
        # ground counts as on it, so a car resting on its suspension doesn't flicker "airborne".
        other = np.where(g1 == self.ground, g2, np.where(g2 == self.ground, g1, -1))
        wheels = np.isin(self.tires, other)
        return wheels, bool(np.any(other == self.chassis))

    def state(self):
        """Oracle state (experts, scoring, rendering; never in the policy's obs)."""
        d = self.d
        R = d.xmat[1].reshape(3, 3)
        pitch, roll = _pitch_roll(R)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        v_world = d.qvel[:3]
        return dict(x=float(d.qpos[0]), y=float(d.qpos[1]), z=float(d.qpos[2]), pitch=float(pitch),
                    roll=float(roll), yaw=float(yaw), up_z=float(R[2, 2]),
                    v_fwd=float(R[:, 0] @ v_world), vx=float(v_world[0]), vz=float(v_world[2]),
                    pitch_rate=float(-d.qvel[4]), yaw_rate=float(d.qvel[5]))

    def obs(self):
        sd = self.d.sensordata
        self.rays = cr.cast_rays(self.m, self.d, self.ray_sites)
        rays = self.rays / cr.RAY_CUTOFF
        w = self.d.qvel[self.spin_dof] * cr.WHEEL_R / 10.0
        return np.concatenate([sd[self.s_accel:self.s_accel + 3] / 9.81, sd[self.s_gyro:self.s_gyro + 3] / 5.0,
                               [w[:2].mean(), w[2:].mean()], rays, self.last_act]).astype(np.float32)

    # ---- one control step ------------------------------------------------------------
    def step(self, action, jump_bonus=10.0, max_steps=1500):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        self.d.ctrl[:4] = a[1]
        self.d.ctrl[4:6] = a[0] * cr.STEER_MAX
        mujoco.mj_step(self.m, self.d, cr.SUBSTEPS)
        self.t += 1
        self.last_act = a
        t = self.t * cr.DT
        wheels, body = self.contacts()
        st = self.state()
        reward = st["x"] - self.x_prev
        self.x_prev = st["x"]

        # airborne bookkeeping
        if not wheels.any():
            if not self.airborne:
                self.airborne, self.takeoff = True, (t, st["x"])
        elif self.airborne:
            self.airborne = False
            air = t - self.takeoff[0]
            self.max_air = max(self.max_air, air)
            if air >= MIN_AIR:
                self.air_log.append(round(air, 3))
                k = self.course.jump_at(self.takeoff[1])
                if k is not None and self.results[k] is None and self.pending is None:
                    self.pending = dict(jump=k, touch=t, air=round(air, 3), all4=False, body=False,
                                        takeoff_x=round(self.takeoff[1], 2), touch_x=round(st["x"], 2),
                                        pitch=round(float(np.degrees(st["pitch"])), 1),
                                        slope=round(float(np.degrees(np.arctan(self.course.slope(st["x"])))), 1),
                                        vz=round(st["vz"], 2), v=round(st["v_fwd"], 2))
        # landing evaluation window
        if self.pending is not None:
            p = self.pending
            p["all4"] |= bool(wheels.all())
            p["body"] |= body
            if t - p["touch"] >= LAND_WINDOW:
                p["upright"] = st["up_z"] >= 0.7
                p["driving"] = st["v_fwd"] >= 0.5
                p["landed"] = bool(p["all4"] and not p["body"] and p["upright"] and p["driving"])
                self.results[p["jump"]] = p
                self.pending = None
                if p["landed"]:
                    reward += jump_bonus

        # endings
        self.slow = self.slow + 1 if st["v_fwd"] < 0.3 else 0
        reason = None
        if st["x"] >= crs.FINISH_X:
            reason = "finished"
        elif st["up_z"] < 0.2:
            reason = "rolled over"
        elif abs(st["y"]) > crs.WIDTH / 2 - 0.2:
            reason = "off the lane"
        elif self.slow >= 100 and self.t > 50:
            reason = "stuck"
        truncated = reason is None and self.t >= max_steps
        if truncated:
            reason = "time limit"
        self.ret += reward
        return reward, reason is not None and not truncated, truncated, reason

    def summary(self, reason):
        if self.pending is not None:   # episode ended inside a landing window
            p = self.pending
            p.update(landed=False, upright=None, driving=None, cut_short=reason)
            self.results[p["jump"]] = p
        jumps = []
        for k, j in enumerate(self.course.jumps):
            r = self.results[k] or dict(jump=k, landed=False, air=None, missed=True)
            jumps.append({**r, "kind": j.kind, "height": round(j.height, 3)})
        return dict(r=round(self.ret, 3), l=self.t, course=self.course.seed, reason=reason,
                    finished=reason == "finished", x=round(self.x_prev, 2),
                    landed=sum(bool(j["landed"]) for j in jumps), n_jumps=len(jumps), jumps=jumps,
                    max_air=round(self.max_air, 3), air_log=self.air_log)


class JumpBatch:
    def __init__(self, n, seed=0, courses="random", max_steps=1500, jump_bonus=10.0, size=1.0, kinds=crs.KINDS):
        rng = np.random.default_rng(seed)
        self.cars = [JumpCar(np.random.default_rng(rng.integers(2**63)), courses, size, kinds) for _ in range(n)]
        if courses != "random":   # spread a fixed course list across the cars
            for i, c in enumerate(self.cars):
                c.courses = list(courses)[i::n] or list(courses)
        self.n, self.max_steps, self.jump_bonus = n, max_steps, jump_bonus
        self.obs_dim = OBS_DIM
        self._obs = None

    def reset(self):
        self._obs = np.stack([c.reset() for c in self.cars])
        return self._obs

    def obs(self):
        return self._obs

    def step(self, actions):
        actions = np.asarray(actions)
        rew = np.zeros(self.n, np.float32)
        term = np.zeros(self.n, bool)
        trunc = np.zeros(self.n, bool)
        infos = [{} for _ in range(self.n)]
        obs = np.empty((self.n, OBS_DIM), np.float32)
        for i, c in enumerate(self.cars):
            rew[i], term[i], trunc[i], reason = c.step(actions[i], self.jump_bonus, self.max_steps)
            if term[i] or trunc[i]:
                infos[i] = dict(terminal_observation=c.obs(), episode=c.summary(reason))
                obs[i] = c.reset()
            else:
                obs[i] = c.obs()
        self._obs = obs
        return obs, rew, term, trunc, infos
