"""A batch of 3D RC cars on jump courses (one MuJoCo model + data per car), with jump scoring.

API like the 2D racers: reset() -> obs (B, D); step(actions (B, 2) in [-1, 1]) ->
obs, reward, terminated, truncated, infos. Finished episodes reset automatically (a new course);
infos[i] then carries "terminal_observation" and "episode" (the summary, incl. per-jump results).

Actions: [steer, throttle]; steer +1 = full left (0.4 rad), throttle +1 = full drive, -1 = full brake/reverse.

Observation (real sensors only, OBS_DIM = 20):
  accel / 9.81 (3, body frame), gyro / 5 (3), wheel speed front/rear / 10 m/s (2),
  8 forward rays (3° up to 50° down) + left/right rays, distance / 6 m (10, 1.0 = nothing within 6 m),
  last action (2).
Optional fan=True adds 6 horizontal rays (20/35/60 deg left and right, level), so the car can see the lane
edges ahead (its heading), before the last action: OBS_DIM + 6.
Optional servo=True adds the steering's actual angle (front wheels, / 0.4 rad), not just the last command: + 1.
Reward option steer_cost: minus steer_cost x |steering command change| each step (owner, 2026-11-13: a
small cost on sawing rather than a hard limit). Reward option finish_bonus: added once at the finish line
(Phase 4 E5, the analogue of Phase 3.5's lap bonus).

A jump (the gate's definition, owner 2026-11-12): all four wheels off the ground for >= 0.3 s.
It is credited to the course jump whose stretch contains the takeoff point (only the first jump
from each ramp counts). It is LANDED if, within 1 s of touchdown, all four wheels are on the
ground together at some moment, the chassis never touches the ground, and at the end of that
second the car is upright (up-axis z >= 0.7) and still driving (forward speed >= 0.5 m/s).
A ramp the car didn't jump (too slow, or it crashed before reaching it) counts as not landed.

Episode ends: finish line (terminated, success), rollover (up-axis z < 0.2), off the lane (centre past the
lane edge, |y| > 1.5 m),
stuck (< 0.3 m/s for 2 s), or the step cap (truncated).

Reward: forward progress in metres + jump_bonus for each landed jump.

loop=True (Phase 4.6): the course is a closed loop with turns (sdc/jumps/loop.py). The car starts just past the
start/finish line and the episode finishes when it crosses the line again (one lap). Position is then measured in
track coordinates: the state's x is metres along the centreline (progress, the reward), y is metres left of it
(off the lane: |y| > 1.5 m), yaw is the heading relative to the track. Sensors and jump scoring are unchanged.
"""

from __future__ import annotations

import numpy as np

import mujoco

from sdc.jumps import car as cr
from sdc.jumps import course as crs
from sdc.jumps import loop as lp
from sdc.jumps import real as rl

OBS_DIM = 20
PARTS_DIM = 6 + 1 + 1 + 64 + 2 + 2   # sensors="parts": IMU, motor speed, TFmini, 8x8 ToF, side ToF x2, last action
TURN_LOOK = (0.0, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0, 13.0)   # m ahead of the car's odometer (turn_map inputs)
TURN_SCALE = 3.0                     # mapped curvature x 3: a 3 m radius reads 1.0
TOF_NORM = 4.0                       # m: 8x8 zones are reported / 4 m (1.0 = no valid return)
ON_GROUND = 0.01   # m: a tire this close to the terrain counts as on the ground
MIN_AIR = 0.3
LAND_WINDOW = 1.0


def _pitch_roll(R):
    pitch = np.arctan2(R[2, 0], np.hypot(R[0, 0], R[1, 0]))   # nose up > 0
    roll = np.arctan2(R[2, 1], R[2, 2])
    return pitch, roll


_MAPS = {}


def _load_map(spec):
    """A track map dict, or a JSON file path holding one (cached)."""
    if spec is None or isinstance(spec, dict):
        return spec
    if spec not in _MAPS:
        import json
        _MAPS[spec] = json.loads(open(spec).read())
    return _MAPS[spec]


class JumpCar:
    """One car on one course."""

    def __init__(self, seed_rng, courses="random", size=1.0, kinds=crs.KINDS, fan=False, servo=False,
                 real=(), sensors="ideal", dr=None, phys_seed=0, attitude=False, track_map=None, loop=False,
                 angled=False, turn_map=None):
        self.real = rl.switches(real)
        # Phase 4.6 E3: the out-lap's map of the turns, {"course|x0": {"ds": m, "kappa": [...]}} (or a JSON path):
        # the curvature the owner's dead reckoning drew along its own odometer. Adds len(TURN_LOOK) inputs, appended
        # last: the mapped curvature at those distances ahead of the car's own odometer (same estimator, running).
        self.turn_map = _load_map(turn_map)
        assert self.turn_map is None or (attitude and sensors == "parts" and loop), "a turn map needs loop + parts + attitude"
        # Phase 4.6 (owner, 2026-11-16): two more VL53L1X on the front corners, 45° forward-left and forward-right,
        # level, 3 m: with the side sensors they tell the car's heading relative to the banks. Two more inputs,
        # appended at the END of the observation, so every earlier slot keeps its place.
        self.angled = angled
        assert not angled or sensors == "parts", "the angled sensors belong to the parts layout"
        # Phase 4.6: loop courses with turns (sdc/jumps/loop.py). The state's x and y are then track coordinates:
        # metres along the centreline from the line (counting on past one lap) and metres left of it.
        self.loop = loop
        self.attitude = attitude   # parts sensors + the owner's filter's pitch and roll as two more inputs (E4)
        # E6: an out-lap map {"course|x0": [[lip odometer m, speed m/s], ...]} (or a JSON path). Adds two inputs:
        # distance to the next mapped lip by the car's own odometer, and that jump's mapped speed.
        self.track_map = _load_map(track_map)
        assert self.track_map is None or (attitude and sensors == "parts"), "a track map needs parts sensors + attitude"
        assert sensors in ("ideal", "noisy", "parts")
        self.sensors, self.dr, self.phys_seed = sensors, (None if dr == "nominal" else dr), phys_seed
        self.legacy = not self.real and sensors == "ideal" and self.dr is None   # Phase 4's car exactly
        self.m = cr.build_model(self.real, lp.TERRAIN if loop else None)
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
        self.fan = fan
        self.ray_sites = np.array([self.m.site(n).id for n in cr.RAY_SITES + (cr.FAN_SITES if fan else [])])
        self.servo = servo
        self.steer_q = np.array([self.m.jnt_qposadr[self.m.joint(f"{w}_steer").id] for w in ("fl", "fr")])
        self.obs_dim = OBS_DIM + (len(cr.FAN_SITES) if fan else 0) + (1 if servo else 0)
        if sensors == "parts":
            self.obs_dim = PARTS_DIM + (2 if attitude else 0) + (2 if self.track_map is not None else 0) + (2 if angled else 0)
            self.obs_dim += len(TURN_LOOK) if self.turn_map is not None else 0
        m = self.m   # nominal values, for per-episode randomization in place
        self._base = dict(body_mass=m.body_mass.copy(), body_inertia=m.body_inertia.copy(),
                          geom_friction=m.geom_friction.copy(), jnt_stiffness=m.jnt_stiffness.copy(),
                          dof_damping=m.dof_damping.copy(), actuator_gear=m.actuator_gear.copy())
        self.susp_j = np.array([m.joint(f"{w}_susp").id for w in cr.WHEELS])
        self.susp_dof = m.jnt_dofadr[self.susp_j]
        self.s_acc_raw = s["accel"][0]
        self.n_motor = 1 if "drive" in self.real else 4
        self.phys = rl.Phys()
        self.tof_dirs = rl.tof_dirs()
        self._scratch = mujoco.MjData(self.m)

    def _apply_phys(self):
        m, b, p = self.m, self._base, self.phys
        m.body_mass[:] = b["body_mass"] * p.mass
        m.body_inertia[:] = b["body_inertia"] * p.mass
        m.geom_friction[:] = b["geom_friction"]
        # MuJoCo uses the larger of the two geoms' friction, so scale the ground too (review, 2026-11-14:
        # scaling only the tires could raise grip but never lower it)
        sel = np.append(self.tires, self.ground)
        m.geom_friction[sel, 0] = b["geom_friction"][sel, 0] * p.friction
        m.jnt_stiffness[:] = b["jnt_stiffness"]
        m.jnt_stiffness[self.susp_j] = b["jnt_stiffness"][self.susp_j] * p.spring
        m.dof_damping[:] = b["dof_damping"]
        m.dof_damping[self.susp_dof] = b["dof_damping"][self.susp_dof] * p.damping
        m.actuator_gear[:] = b["actuator_gear"]
        m.actuator_gear[:self.n_motor, 0] = b["actuator_gear"][:self.n_motor, 0] * p.motor
        mujoco.mj_setConst(m, self._scratch)   # constants derived from the masses (it overwrites the data it's given)

    # ---- episode ---------------------------------------------------------------------
    def reset(self, course_seed=None, x0=0.0):
        if course_seed is None:
            if self.courses == "random":
                course_seed = int(self.rng.integers(0, 2**31 - 1))
            else:
                course_seed = self.courses[self._course_i % len(self.courses)]
                self._course_i += 1
        if self.loop:
            self.course = lp.make(course_seed, kinds=self.kinds if "double" in self.kinds else lp.KINDS, size=self.size)
        else:
            self.course = crs.make(course_seed, kinds=self.kinds, size=self.size)
        self.m.hfield_data[:] = self.course.hfield().ravel()
        if not self.legacy:   # physics and sensor noise drawn from (course, start, phys_seed): batching-independent
            self.prng = np.random.default_rng([int(course_seed), int(round(x0 * 1000)) + 10_000, int(self.phys_seed)])
            self.phys = rl.draw(self.dr, self.prng)
            self._apply_phys()
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0] = crs.START_X + x0      # x0: start offset (evaluation repeats a course from a few starts)
        self.d.qpos[2] = self.course.height(crs.START_X + x0) + 0.17
        if self.loop:   # on the centreline, START_X + x0 past the line, facing along the track
            X, Y, th = self.course.pose(crs.START_X + x0)
            self.d.qpos[:2] = X, Y
            self.d.qpos[3:7] = np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)
            self._si, self._s = int(round((crs.START_X + x0) / self.course.ds)), crs.START_X + x0
        mujoco.mj_forward(self.m, self.d)
        mujoco.mj_step(self.m, self.d, int(0.2 / cr.TIMESTEP))     # settle on the suspension (0.2 s)
        self.v_prev = self.d.qvel[:3].copy()
        self.t = 0
        self.last_act = np.zeros(2, np.float32)
        self.ret = 0.0
        self.x_prev = self.state()["x"]
        self.slow = 0
        self.airborne = False
        self.takeoff = None          # (t, x)
        self.body_hit = False        # chassis touched the ground since takeoff (checked every physics step)
        self.pending = None          # landing under evaluation
        self.results = [None] * len(self.course.jumps)   # per course jump: dict or None
        self.air_log = []            # every airborne spell >= MIN_AIR
        self.max_air = 0.0
        if not self.legacy:
            self.imu = rl.IMU(self.prng, self.phys)
            self.sub = 0                         # physics steps since the start
            self.queue = []                      # (physics step it takes effect, action): command latency
            self.applied = np.zeros(2, np.float32)   # the command the actuators are acting on now
            self.servo_set = 0.0                 # the servo's current target angle (rad), speed-limited
            self.n_clipped = 0                   # control steps with an accelerometer sample clipped at 16 g
            self.tof_next, self.tof = 0.0, np.full(64, TOF_NORM)
            self._calibrate()
            if self.track_map is not None:
                self.map = sorted(self.track_map.get(f"{course_seed}|{x0:.2f}", []))
                self.odo, self.v_ground, self.free = 0.0, 0.0, False
                self.map_next_speed = None
            if self.attitude:
                from sdc.phase45 import my_filter as mf   # the owner's filter, run on the car's own sensors
                self.filt = mf.AttitudeFilter(telemetry=True)
                self.filt.reset(self.still_gyro, self.still_acc)
            if self.turn_map is not None:
                from sdc.phase46 import my_deadreckoning as mdr   # the owner's dead reckoning: the car's odometer
                self.dead = mdr.DeadReckoning(start_x=crs.START_X)
                self.dead.reset(self.still_gyro)
                m = self.turn_map.get(f"{course_seed}|{x0:.2f}") or {}
                self.tmap = (float(m.get("ds", 0.25)), np.asarray(m.get("kappa", []), float))
                self.kick_marks, self._kick_run, self._kick_last = np.asarray(m.get("kickers", []), float), 0, -1e9
                self.resyncs = []
            self._sample_imu()
            self._obs = self._read()
            return self._obs
        return self.obs()

    def contacts(self):
        """(wheels on the ground (4,), chassis touching the ground).

        A wheel counts as on the ground when its tire is within ON_GROUND (1 cm) of the terrain
        (mj_geomDistance), so a car resting on its suspension doesn't flicker "airborne" from contact
        chatter. (Until 2026-11-13 this used a contact margin on the tires, which in MuJoCo 4 also
        made them float 1 cm above the ground: the review found it.)"""
        wheels = np.array([mujoco.mj_geomDistance(self.m, self.d, int(g), self.ground, 0.05, None) < ON_GROUND
                           for g in self.tires])
        c = self.d.contact
        n = self.d.ncon
        g1, g2 = c.geom1[:n], c.geom2[:n]
        body = bool(np.any(((g1 == self.chassis) & (g2 == self.ground)) | ((g2 == self.chassis) & (g1 == self.ground))))
        return wheels, body

    def _chassis_contact(self):
        n = self.d.ncon
        g1, g2 = self.d.contact.geom1[:n], self.d.contact.geom2[:n]
        return bool(np.any(((g1 == self.chassis) & (g2 == self.ground)) | ((g2 == self.chassis) & (g1 == self.ground))))

    def state(self):
        """Oracle state (experts, scoring, rendering; never in the policy's obs)."""
        d = self.d
        R = d.xmat[1].reshape(3, 3)
        pitch, roll = _pitch_roll(R)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        v_world = d.qvel[:3]
        if self.loop:   # track coordinates
            c = self.course
            s, off, th, self._si = c.project(d.qpos[0], d.qpos[1], self._si)
            self._s += (s - self._s + c.length / 2) % c.length - c.length / 2    # keeps counting past the line
            kap = float(c.curvature(s))
            vs = float(np.cos(th) * v_world[0] + np.sin(th) * v_world[1])
            return dict(x=float(self._s), y=off, z=float(d.qpos[2]), pitch=float(pitch), roll=float(roll),
                        yaw=float((yaw - th + np.pi) % (2 * np.pi) - np.pi), up_z=float(R[2, 2]),
                        v_fwd=float(R[:, 0] @ v_world), vx=vs, vz=float(v_world[2]), pitch_rate=float(-d.qvel[4]),
                        yaw_rate=float(d.qvel[5]) - vs * kap, kappa=kap, X=float(d.qpos[0]), Y=float(d.qpos[1]),
                        heading=float(yaw))
        return dict(x=float(d.qpos[0]), y=float(d.qpos[1]), z=float(d.qpos[2]), pitch=float(pitch),
                    roll=float(roll), yaw=float(yaw), up_z=float(R[2, 2]),
                    v_fwd=float(R[:, 0] @ v_world), vx=float(v_world[0]), vz=float(v_world[2]),
                    pitch_rate=float(-d.qvel[4]), yaw_rate=float(d.qvel[5]))

    def obs(self):
        if not self.legacy:
            return self._obs
        return self._ideal_obs()

    def _ideal_obs(self):
        sd = self.d.sensordata
        self.rays = cr.cast_rays(self.m, self.d, self.ray_sites)
        rays = self.rays / cr.RAY_CUTOFF
        w = self.d.qvel[self.spin_dof] * cr.WHEEL_R / 10.0
        # accelerometer averaged over the 20 ms control step (a real IMU low-pass filters; a single
        # 2 ms sample was dominated by contact chatter): (dv/dt + gravity) in the body frame
        R = self.d.xmat[1].reshape(3, 3)
        acc = R.T @ ((self.d.qvel[:3] - self.v_prev) / cr.DT + np.array([0.0, 0.0, 9.81]))
        servo = [self.d.qpos[self.steer_q].mean() / cr.STEER_MAX] if self.servo else []   # actual wheel angle
        return np.concatenate([acc / 9.81, sd[self.s_gyro:self.s_gyro + 3] / 5.0,
                               [w[:2].mean(), w[2:].mean()], rays, servo, self.last_act]).astype(np.float32)

    # ---- realistic sensors and actuators (Phase 4.5) ----------------------------------
    def _mean_wheel_speed(self):
        return float(self.d.qvel[self.spin_dof].mean() * cr.WHEEL_R)

    def _sample_imu(self):
        sd = self.d.sensordata
        frac = min(abs(self._mean_wheel_speed()) / cr.TOP_SPEED, 1.0)
        self.imu.sample(sd[self.s_acc_raw:self.s_acc_raw + 3].copy(), sd[self.s_gyro:self.s_gyro + 3].copy(), frac,
                        cr.TIMESTEP)

    def _calibrate(self):
        """IMU readings while the car sits still before the start (0.2 s, 10 control-step reads, after 0.5 s more
        settling, held still), for a gyro-bias
        calibration like a real car's on power-up. The physics state is restored afterwards, so the run itself
        starts exactly as before (the car was resting anyway)."""
        d = self.d
        saved = (d.qpos.copy(), d.qvel.copy(), d.act.copy(), d.qacc_warmstart.copy(), d.time, d.ctrl.copy())
        d.ctrl[:] = 0.0
        # let the soft suspension stop bouncing first (0.5 s, unrecorded): a still car, as on a real power-up
        mujoco.mj_step(self.m, d, int(0.5 / cr.TIMESTEP))
        self.imu.clear()
        acc, gyro = [], []
        for _ in range(10):
            for _ in range(cr.SUBSTEPS):
                d.qvel[:] = 0.0          # held still (on a bumpy start the car would otherwise creep)
                mujoco.mj_step(self.m, d, 1)
                self._sample_imu()
            a, g = self.imu.read()
            acc.append(a)
            gyro.append(g)
        self.still_acc, self.still_gyro = np.array(acc), np.array(gyro)
        d.qpos[:], d.qvel[:], d.act[:], d.qacc_warmstart[:], d.time, d.ctrl[:] = saved
        mujoco.mj_forward(self.m, d)

    def _cast(self, origin_body, dirs_body, cutoff):
        R = self.d.xmat[1].reshape(3, 3)
        o = self.d.xpos[1] + R @ np.asarray(origin_body, float)
        if self.loop:   # the loops' heightfield is large: march the height map instead of MuJoCo's ray cast
            return lp.cast(self.course, o, dirs_body @ R.T, cutoff)
        vec = (dirs_body @ R.T).ravel()
        n = len(dirs_body)
        dist, gid = np.empty(n), np.empty(n, np.int32)
        mujoco.mj_multiRay(self.m, self.d, o, vec, np.array([1, 0, 0, 0, 0, 0], np.uint8), True, -1, gid, dist,
                           None, n, cutoff)
        dist[(dist < 0) | (dist > cutoff)] = cutoff
        return dist

    def _read(self):
        """The policy's observation from the realistic sensors (end of a control step)."""
        if self.sensors == "ideal":   # Phase 4's perfect sensors on the realistic car
            self.imu.clear()
            return self._ideal_obs()
        rng, k = self.prng, self.phys.noise
        acc, gyro = self.imu.read()
        spd = rl.motor_speed(rng, self._mean_wheel_speed()) / 10.0
        if self.sensors == "noisy":   # Phase 4's layout
            self.rays = cr.cast_rays(self.m, self.d, self.ray_sites)
            rays = rl.ranges(rng, self.rays, cr.RAY_CUTOFF, k) / cr.RAY_CUTOFF
            servo = [float(self.last_act[0])] if self.servo else []   # no position feedback: the command
            return np.concatenate([acc / 9.81, gyro / 5.0, [spd, spd], rays, servo, self.last_act]).astype(np.float32)
        # parts: TFmini-S (3° down), VL53L5CX 8x8 at 15 Hz (sample-and-hold), VL53L1X left/right
        t = self.t * cr.DT
        if t >= self.tof_next - 1e-9:   # frames on a 15 Hz schedule (sample-and-hold between them)
            self.tof_next += 1.0 / rl.TOF_HZ
            r = self.phys.tof_range
            z = rl.ranges(rng, self._cast((0.235, 0, 0.02), self.tof_dirs, r), r, k, rl.TOF_DROPOUT)
            z[z >= r] = TOF_NORM
            self.tof = z
        tf_dir = np.array([[np.cos(np.deg2rad(3)), 0, -np.sin(np.deg2rad(3))]])
        tf = rl.ranges(rng, self._cast((0.235, 0, 0.02), tf_dir, rl.TFMINI_MAX), rl.TFMINI_MAX, k)
        side = rl.ranges(rng, self._cast((0, 0, 0.02), np.array([[0, 1.0, 0], [0, -1.0, 0]]), rl.SIDE_MAX),
                         rl.SIDE_MAX, k)
        att, mp = [], []
        if self.attitude:
            att = np.array(self.filt.update(acc, gyro, speed=spd * 10.0))
            if self.track_map is not None:
                mp = self._map_inputs(acc, spd * 10.0, att[0])
            att = att / 0.5   # rad / 0.5
        tm = []
        if self.turn_map is not None:
            self.dead.update(gyro, spd * 10.0, float(att[0]) * 0.5, float(att[1]) * 0.5, acc=acc, steer=float(self.last_act[0]))
            self._resync(float(att[0]) * 0.5, acc)
            tm = [self.map_kappa(d) * TURN_SCALE for d in TURN_LOOK]
        ang = []
        if self.angled:
            c45 = np.cos(np.pi / 4)
            ang = [rl.ranges(rng, self._cast((0.2, y, 0.02), np.array([[c45, sy * c45, 0.0]]), rl.SIDE_MAX),
                             rl.SIDE_MAX, k)[0] / rl.SIDE_MAX for y, sy in ((0.08, 1.0), (-0.08, -1.0))]
        return np.concatenate([acc / 9.81, gyro / 5.0, [spd], tf / rl.TFMINI_MAX, self.tof / TOF_NORM,
                               side / rl.SIDE_MAX, self.last_act, att, mp, ang, tm]).astype(np.float32)

    def _resync(self, pitch, acc):
        """Re-sync the odometer at a kicker: the same landmark the out-lap mapped (maps.kickers): the filter's pitch
        climbing through 12° on the ground. The odometer snaps to the nearest mapped kicker within 8 m."""
        on = pitch > np.deg2rad(12) and np.linalg.norm(acc) / 9.81 > 0.4
        self._kick_run = self._kick_run + 1 if on else 0
        if self._kick_run == 3 and len(self.kick_marks) and self.dead.s - self._kick_last > 8.0:
            lap = self.tmap[0] * len(self.tmap[1])
            d = (self.kick_marks - self.dead.s + lap / 2) % lap - lap / 2
            k = int(np.argmin(np.abs(d)))
            if abs(d[k]) < 8.0:
                self.resyncs.append(round(float(d[k]), 2))
                self.dead.s += float(d[k])
            self._kick_last = self.dead.s

    def map_kappa(self, ahead=0.0):
        """The mapped curvature `ahead` metres past the car's own odometer (0 if this run has no map)."""
        ds, kap = self.tmap
        if not len(kap):
            return 0.0
        return float(kap[int(round((self.dead.s + ahead) / ds)) % len(kap)])

    def _map_inputs(self, acc, speed, pitch):
        """Odometer (motor speed, held in free fall, horizontal via the filter's pitch), re-synced to the nearest mapped
        lip at each takeoff; returns [distance to the next mapped lip / 10, its mapped speed / 10]."""
        free = np.linalg.norm(acc) / 9.81 < 0.4
        if not free:
            self.v_ground = speed
        if free and not self.free and self.map:   # takeoff: snap to the nearest mapped lip within 3 m
            lips = np.array([m[0] for m in self.map])
            k = int(np.argmin(np.abs(lips - self.odo)))
            if abs(lips[k] - self.odo) < 3.0:
                self.odo = float(lips[k])
        self.free = free
        self.odo += self.v_ground * cr.DT * float(np.cos(pitch))
        ahead = [m for m in self.map if m[0] > self.odo - 0.3]
        if not ahead:
            self.map_next_speed = None
            return [3.0, 0.74]
        self.map_next_speed = float(ahead[0][1])
        return [float(np.clip((ahead[0][0] - self.odo) / 10.0, -0.5, 3.0)), ahead[0][1] / 10.0]

    def _actuate(self):
        """Set the controls for one physics step from the latest command that has arrived."""
        while self.queue and self.queue[0][0] <= self.sub:
            self.applied = self.queue.pop(0)[1]
        target = self.applied[0] * cr.STEER_MAX   # float32, exactly as Phase 4 computes it (chaos amplifies 1e-9)
        if "servo" in self.real:   # speed-limited, with a deadband
            err = target - self.servo_set
            if abs(err) > self.phys.deadband:
                rate = np.deg2rad(60) / self.phys.servo_s60
                self.servo_set += float(np.clip(err, -rate * cr.TIMESTEP, rate * cr.TIMESTEP))
        else:
            self.servo_set = target
        u = float(self.applied[1])
        if "battery" in self.real:
            u *= self.phys.charge * (1.0 - self.phys.sag * abs(u))
        if "esc" in self.real:   # physics review: a hobby ESC's brake, not reverse torque (one motor: "drive")
            raw = float(self.applied[1])
            w = float(self.d.qvel[self.spin_dof].mean()) * cr.WHEEL_R / cr.motor(self.real)[1]   # motor speed / top
            if raw > 0.05:
                u = u - w                                    # drive: torque falls with back-EMF, as the damper did
            elif raw < -0.05:
                u = -abs(u) * float(np.tanh(w * cr.motor(self.real)[1] / 0.3))   # opposes rotation, 0 at standstill
            else:
                u = -0.1 * w                                 # neutral: coast with a light drag brake
            u = float(np.clip(u, -1.0, 1.0))
        self.d.ctrl[:self.n_motor] = u
        self.d.ctrl[self.n_motor:self.n_motor + 2] = self.servo_set

    def _physics(self, a):
        """One control step of physics with the realistic actuators and the IMU sampled every 2 ms."""
        delay = self.phys.delay if "delay" in self.real else 0.0
        lag = max(0, int(round((delay + self.prng.uniform(-0.001, 0.001) * (delay > 0)) / cr.TIMESTEP)))
        self.queue.append((self.sub + lag, a))
        check = self.airborne or self.pending is not None
        self.v_prev = self.d.qvel[:3].copy()
        for _ in range(cr.SUBSTEPS):
            self._actuate()
            mujoco.mj_step(self.m, self.d, 1)
            self.sub += 1
            if check:
                self.body_hit |= self._chassis_contact()
            self._sample_imu()
        self.n_clipped += int(self.imu.clipped)

    # ---- one control step ------------------------------------------------------------
    def step(self, action, jump_bonus=10.0, max_steps=1500, steer_cost=0.0, finish_bonus=0.0):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        d_steer = abs(float(a[0]) - float(self.last_act[0]))
        if not self.legacy:
            return self._step_tail(a, d_steer, jump_bonus, max_steps, steer_cost, finish_bonus)
        self.d.ctrl[:4] = a[1]
        self.d.ctrl[4:6] = a[0] * cr.STEER_MAX
        v0 = self.d.qvel[:3].copy()
        if self.airborne or self.pending is not None:
            # In the air and in a landing window, check the chassis after every 2 ms physics step: a bumper tap
            # can come and go inside one 20 ms control step (pre-gate review, 2026-11-13). Same physics as
            # one mj_step of 10 substeps.
            for _ in range(cr.SUBSTEPS):
                mujoco.mj_step(self.m, self.d, 1)
                self.body_hit |= self._chassis_contact()
        else:
            mujoco.mj_step(self.m, self.d, cr.SUBSTEPS)
        self.v_prev = v0
        return self._step_tail(a, d_steer, jump_bonus, max_steps, steer_cost, finish_bonus, physics_done=True)

    def _step_tail(self, a, d_steer, jump_bonus, max_steps, steer_cost, finish_bonus, physics_done=False):
        if not physics_done:
            self._physics(a)
        self.t += 1
        self.last_act = a
        t = self.t * cr.DT
        wheels, body = self.contacts()
        st = self.state()
        reward = st["x"] - self.x_prev - steer_cost * d_steer
        self.x_prev = st["x"]

        # airborne bookkeeping
        if not wheels.any():
            if not self.airborne:
                self.airborne, self.takeoff = True, (t, st["x"])
                if self.pending is None:
                    self.body_hit = False
        elif self.airborne:
            self.airborne = False
            air = t - self.takeoff[0]
            self.max_air = max(self.max_air, air)
            if air >= MIN_AIR:
                self.air_log.append(round(air, 3))
                k = self.course.jump_at(self.takeoff[1])
                if k is not None and self.results[k] is None and self.pending is None:
                    self.pending = dict(jump=k, touch=t, air=round(air, 3), all4=False, body=self.body_hit,
                                        takeoff_x=round(self.takeoff[1], 2), touch_x=round(st["x"], 2),
                                        pitch=round(float(np.degrees(st["pitch"])), 1),
                                        slope=round(float(np.degrees(np.arctan(self.course.slope(st["x"])))), 1),
                                        vz=round(st["vz"], 2), v=round(st["v_fwd"], 2))
        # landing evaluation window
        if self.pending is not None:
            p = self.pending
            p["all4"] |= bool(wheels.all())
            p["body"] |= body or self.body_hit
            if t - p["touch"] >= LAND_WINDOW:
                p["upright"] = st["up_z"] >= 0.7
                p["driving"] = st["v_fwd"] >= 0.5
                p["landed"] = bool(p["all4"] and not p["body"] and p["upright"] and p["driving"])
                self.results[p["jump"]] = p
                self.pending = None
                self.body_hit = False
                if p["landed"]:
                    reward += jump_bonus

        # endings
        self.slow = self.slow + 1 if st["v_fwd"] < 0.3 else 0
        reason = None
        if st["x"] >= (self.course.finish if self.loop else crs.FINISH_X):
            reason = "finished"
            reward += finish_bonus
        elif st["up_z"] < 0.2:
            reason = "rolled over"
        elif abs(st["y"]) > crs.DRIVE_W / 2:     # centre past the lane edge: its outer wheels are on the bank
            reason = "off the lane"
        elif self.slow >= 100 and self.t > 50:
            reason = "stuck"
        truncated = reason is None and self.t >= max_steps
        if truncated:
            reason = "time limit"
        self.ret += reward
        if not self.legacy:
            self._obs = self._read()
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
                    max_air=round(self.max_air, 3), air_log=self.air_log,
                    **({} if self.legacy else dict(phys={k: round(v, 4) for k, v in vars(self.phys).items()},
                                                   clipped=self.n_clipped)))


class JumpBatch:
    def __init__(self, n, seed=0, courses="random", max_steps=1500, jump_bonus=10.0, size=1.0, kinds=crs.KINDS,
                 fan=False, servo=False, steer_cost=0.0, finish_bonus=0.0, real=(), sensors="ideal", dr=None,
                 phys_seed=0, attitude=False, track_map=None, loop=False, angled=False, turn_map=None):
        rng = np.random.default_rng(seed)
        self.cars = [JumpCar(np.random.default_rng(rng.integers(2**63)), courses, size, kinds, fan, servo,
                             real, sensors, dr, phys_seed, attitude, track_map, loop, angled, turn_map) for _ in range(n)]
        self.steer_cost, self.finish_bonus = steer_cost, finish_bonus
        if courses != "random":   # spread a fixed course list across the cars
            for i, c in enumerate(self.cars):
                c.courses = list(courses)[i::n] or list(courses)
        self.n, self.max_steps, self.jump_bonus = n, max_steps, jump_bonus
        self.obs_dim = self.cars[0].obs_dim
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
        obs = np.empty((self.n, self.obs_dim), np.float32)
        for i, c in enumerate(self.cars):
            rew[i], term[i], trunc[i], reason = c.step(actions[i], self.jump_bonus, self.max_steps, self.steer_cost, self.finish_bonus)
            if term[i] or trunc[i]:
                infos[i] = dict(terminal_observation=c.obs(), episode=c.summary(reason))
                obs[i] = c.reset()
            else:
                obs[i] = c.obs()
        self._obs = obs
        return obs, rew, term, trunc, infos
