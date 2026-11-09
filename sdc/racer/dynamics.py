"""Phase 3 racer: a dynamic bicycle model with tire slip, dirt patches, delays and sensor noise.

Same tracks, rays, rewards and episode rules as the Phase 1 racer (sim.py), but the
car is no longer on rails. It has a body with mass and yaw inertia, and two tires
(front axle, rear axle) that push sideways only when they slip.

Tires (one per axle). The slip angle α is the angle between where the wheel points
and where it is actually moving. Sideways force grows with α, peaks at α ≈ 6–7°,
then falls to about 70% of the peak as the tire slides (simplified Pacejka "magic
formula"). The peak is μ·(load on that axle): μ = GRIP0 × grip. Throttle and brake
use up the same grip (friction ellipse): a tire that is braking hard has less left
for turning. If the rear tires saturate first, the car oversteers (the tail slides
out, it can spin). If the fronts saturate first, it understeers (it ploughs wide).

Drive and brakes. Throttle pushes with a fixed motor force (MASS0·ACCEL, limited
to POWER / speed at speed), split 50/50 front/rear (all-wheel drive). Brakes push
with a fixed force (MASS0·BRAKE = 12 m/s² worth), split 50/50, which is more than
the tires can hold: full brake locks the wheels, and a locked tire can't steer.
Both are capped by grip. A heavier car therefore accelerates and brakes more slowly
and needs more slip to turn. Top speed V_MAX, no reverse.

Below about 3 m/s slip angles are ill-defined, so the model blends into the
Phase 1 kinematic model (no slip) at walking pace.

Action (each in [-1, 1]):
  steer:    front wheel angle as a fraction of STEER_MAX (30°), + = left.
            Unlike Phase 1, nothing stops you from turning hard at speed: you can spin.
  throttle: + drive, − brake

Observation (15 numbers):
  11 rays / RAY_RANGE, forward speed / V_MAX, last steer command, last throttle
  command, yaw rate (gyro) / YAW_SCALE. Sensor noise, if any, is added to the rays,
  speed and yaw rate. The car is NOT told the grip, the mass, the delay, or where
  the dirt is: it has to feel them.

Physics randomization (per episode, see PhysicsRange): grip multiplier, mass
multiplier, actuation delay (seconds between choosing an action and the wheels
getting it), and sensor noise (std of the noise as a fraction of each sensor's full
scale). Dirt patches: 2–4 stretches of 10–30 m per track at DIRT_GRIP (65%) × grip,
placed from the track seed, so they are the same every time on a given track.

Realism level 2 (default from Phase 3 E4; E1-E3 used level 1, without these):
  - steering servo: the wheels turn toward the commanded angle at a limited rate, SERVO_TIME
    (0.4 s, a loaded RC servo) from full left to full right
  - locked/spinning wheels: asking a tire for more braking or drive force than its peak makes
    it slide, and it then delivers only SLIDE_FRAC (75%) of the peak, with nothing for turning
  - tire relaxation: sideways force builds up over RELAX_LEN (0.5 m) of rolling after the slip
    angle changes, instead of instantly, so very fast steering changes do little

Integration: 10 substeps of 10 ms per 0.1 s control step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sdc.racer import sim
from sdc.racer import track as trk

G = 9.81
GRIP0 = 0.85            # nominal tire friction μ: 0.85 g ≈ Phase 1's 8 m/s² cornering limit
MASS0 = 1200.0          # kg (nominal)
LF = LR = sim.WHEELBASE / 2
IZ_PER_KG = LF * LR     # yaw inertia / mass, m²
C_ALPHA = 8.0e4         # cornering stiffness per axle at small slip, N/rad (a tire property: fixed)
PAC_C = 1.5             # magic-formula shape: past the peak, force drops toward sin(1.5·π/2) ≈ 71%
DRIVE_SPLIT = (0.5, 0.5)
POWER = 100e3           # W: full drive force up to about 14 m/s, then force falls as power / speed
BRAKE_SPLIT = (0.5, 0.5)
STEER_MAX = sim.STEER_MAX
SUBSTEPS = 10
V_KIN, V_DYN = 1.5, 3.0  # below V_KIN purely kinematic, above V_DYN purely dynamic
YAW_SCALE = 2.0          # rad/s that maps to 1.0 in the observation
DIRT_GRIP = 0.65
MAX_DELAY_STEPS = 3      # delays up to 0.3 s are supported
OBS_DIM = sim.N_RAYS + 4
# Optional extra sensors (appended after the 15 standard numbers, in this order; Phase 3 E5):
#   "velocity"  sideways velocity of the car's centre in its own frame, / VY_SCALE: with forward speed,
#               the velocity vector, so the angle between where it points and where it goes (a real
#               car needs an optical-flow sensor or an estimator for this)
#   "accel"     accelerometer: forward and sideways acceleration the body feels, / G, averaged over
#               the step (a cheap IMU chip)
SENSORS = {"velocity": 1, "accel": 2}
# history=k (Phase 3 E7): also append the car's own-motion readings (speed, last commands, gyro, and any
# extra sensors) from the previous k-1 steps, newest first, so it can tell how the car has been
# responding to its commands lately, e.g. to estimate the grip. Zeros before the episode's start.
VY_SCALE = 5.0
# realism=2 (Phase 3 E4 on; E1-E3 ran with realism=1, which has none of these):
SERVO_TIME = 0.4         # s from full left to full right lock (a loaded RC servo; 0.2 s unloaded)
SERVO_RATE = 2 * STEER_MAX / SERVO_TIME
RELAX_LEN = 0.5          # m a tire rolls before its sideways force catches up with a new slip angle
SLIDE_FRAC = 0.75        # a locked or spinning wheel's front-to-back grip, as a fraction of the peak


@dataclass(frozen=True)
class PhysicsRange:
    """Uniform ranges for each episode's physics. (x, x) = fixed."""
    grip: tuple[float, float] = (1.0, 1.0)
    mass: tuple[float, float] = (1.0, 1.0)
    delay: tuple[float, float] = (0.0, 0.0)    # seconds
    noise: tuple[float, float] = (0.0, 0.0)    # std, fraction of full scale

    def to_dict(self):
        return dict(grip=self.grip, mass=self.mass, delay=self.delay, noise=self.noise)


NOMINAL = PhysicsRange()
TRAIN_DR = PhysicsRange(grip=(0.7, 1.1), mass=(0.8, 1.2), delay=(0.0, 0.06), noise=(0.0, 0.02))
GATE_DR = PhysicsRange(grip=(0.5, 1.3), mass=(0.65, 1.35), delay=(0.0, 0.12), noise=(0.0, 0.04))
RANGES = dict(nominal=NOMINAL, train=TRAIN_DR, gate=GATE_DR)


def physics_range(spec) -> PhysicsRange:
    if isinstance(spec, PhysicsRange):
        return spec
    if isinstance(spec, str):
        return RANGES[spec]
    return PhysicsRange(**{k: tuple(v) if isinstance(v, (list, tuple)) else (v, v) for k, v in spec.items()})


def dirt_mask(seed: int) -> np.ndarray:
    """(N_POINTS,) bool: which centerline points are dirt. Fixed per track seed."""
    t = trk.generate(seed)
    rng = np.random.default_rng([seed, 31337])
    mask = np.zeros(trk.N_POINTS, bool)
    for _ in range(int(rng.integers(2, 5))):
        start = int(rng.integers(trk.N_POINTS))
        n = int(np.ceil(rng.uniform(10, 30) / t.ds))
        mask[(start + np.arange(n)) % trk.N_POINTS] = True
    return mask


def tire_force(alpha, peak):
    """Sideways tire force for slip angle alpha (rad) and peak force mu·Fz (N). Opposes the slip."""
    B = C_ALPHA / (PAC_C * np.maximum(peak, 1e-6))
    return -peak * np.sin(PAC_C * np.arctan(B * alpha))


class DynamicRacerBatch(sim.RacerBatch):
    """B cars with tire dynamics. Same interface as sim.RacerBatch, observation has OBS_DIM numbers.

    physics: PhysicsRange, a name in RANGES, or a dict of ranges, drawn per episode.
    dirt: put the track's dirt patches down (True) or not.
    """

    def __init__(self, n: int, physics="nominal", dirt: bool = True, realism: int = 2, sensors=(), history: int = 1,
                 **kw):
        assert realism in (1, 2), realism
        self.sensors = tuple(sensors or ())
        assert all(x in SENSORS for x in self.sensors), self.sensors
        base_dim = OBS_DIM + sum(SENSORS[x] for x in self.sensors)
        self.proprio = np.r_[sim.N_RAYS:base_dim]          # everything but the rays
        self.history = int(history)
        self.phist = np.zeros((n, max(self.history - 1, 0), len(self.proprio)), np.float32)
        self.obs_dim = base_dim + self.phist.shape[1] * self.phist.shape[2]
        self._acc_sum = np.zeros((n, 2))
        self.accel = np.zeros((n, 2))   # body-frame specific force (what an accelerometer reads), m/s², step mean
        self.realism = realism
        self.wheel = np.zeros(n)                             # actual front wheel angle (servo), rad
        self.fy = np.zeros((n, 2))                           # current sideways tire forces (relaxation), N
        self.locked = np.zeros((n, 2), bool)                 # axle sliding front-to-back (locked or spinning)
        self.physics = physics_range(physics)
        self.dirt = dirt
        self._dirt_cache: dict[int, np.ndarray] = {}
        self.vx = np.zeros(n)
        self.vy = np.zeros(n)
        self.yaw_rate = np.zeros(n)
        self.grip = np.ones(n)
        self.mass = np.ones(n)
        self.delay = np.zeros(n)
        self.noise = np.zeros(n)
        self.dirt_pts = np.zeros((n, trk.N_POINTS), bool)
        self.hist = np.zeros((n, MAX_DELAY_STEPS + 1, 2))   # hist[:, k] = command issued k steps ago
        self.slip = np.zeros((n, 2))                         # last slip angles front, rear (rad)
        self._obs = np.zeros((n, self.obs_dim), np.float32)
        super().__init__(n, **kw)
        self.noise_rng = np.random.default_rng([kw.get("seed", 0), 7])

    def _dirt(self, seed):
        if seed not in self._dirt_cache:
            self._dirt_cache[seed] = dirt_mask(seed)
        return self._dirt_cache[seed]

    def _reset_car(self, i: int):
        super()._reset_car(i)
        p = self.physics
        u = self.rng.random(4)
        self.grip[i] = p.grip[0] + u[0] * (p.grip[1] - p.grip[0])
        self.mass[i] = p.mass[0] + u[1] * (p.mass[1] - p.mass[0])
        self.delay[i] = p.delay[0] + u[2] * (p.delay[1] - p.delay[0])
        self.noise[i] = p.noise[0] + u[3] * (p.noise[1] - p.noise[0])
        self.dirt_pts[i] = self._dirt(int(self.seeds[i])) if self.dirt else False
        self.vx[i] = self.vy[i] = self.yaw_rate[i] = 0.0
        self.accel[i] = 0.0
        self.phist[i] = 0.0
        self.hist[i] = 0.0
        self.slip[i] = 0.0
        self.wheel[i] = 0.0
        self.fy[i] = 0.0
        self.locked[i] = False

    def set_physics(self, i: int, grip=None, mass=None, delay=None, noise=None):
        """Override one car's drawn physics (robustness sweeps). Call after reset()."""
        for name, val in dict(grip=grip, mass=mass, delay=delay, noise=noise).items():
            if val is not None:
                getattr(self, name)[i] = val
        self._make_obs()

    def reset(self) -> np.ndarray:
        super().reset()
        self._make_obs()
        out = self.obs()
        self._push_history()
        return out

    def _push_history(self):
        """Remember this step's own-motion readings for the next observations."""
        if self.phist.shape[1]:
            cur = self._obs[:, self.proprio]
            self.phist = np.concatenate([cur[:, None], self.phist[:, :-1]], axis=1)

    def _make_obs(self):
        sig = self.noise[:, None]
        rays = self.rays / sim.RAY_RANGE + sig * self.noise_rng.standard_normal((self.n, sim.N_RAYS))
        spd = self.vx / sim.V_MAX + self.noise * self.noise_rng.standard_normal(self.n)
        yaw = self.yaw_rate / YAW_SCALE + self.noise * self.noise_rng.standard_normal(self.n)
        parts = [np.clip(rays, 0, 1.2), np.clip(spd, -0.2, 1.2)[:, None], self.hist[:, 0], np.clip(yaw, -3, 3)[:, None]]
        for name in self.sensors:
            if name == "velocity":
                vy = self.vy / VY_SCALE + self.noise * self.noise_rng.standard_normal(self.n)
                parts.append(np.clip(vy, -3, 3)[:, None])
            else:
                acc = self.accel / G + self.noise[:, None] * self.noise_rng.standard_normal((self.n, 2))
                parts.append(np.clip(acc, -3, 3))
        if self.phist.shape[1]:
            parts.append(self.phist.reshape(self.n, -1))
        self._obs = np.concatenate(parts, axis=1).astype(np.float32)

    def obs(self) -> np.ndarray:
        return self._obs.copy()

    @property
    def surface_grip(self):
        """Current μ under the car (dirt patches lower it)."""
        on_dirt = self.dirt_pts[np.arange(self.n), self.idx]
        return GRIP0 * self.grip * np.where(on_dirt, DIRT_GRIP, 1.0)

    # ---- dynamics -----------------------------------------------------------------------
    def _substep(self, cmd, dt):
        m = MASS0 * self.mass
        iz = IZ_PER_KG * m
        mu = self.surface_grip
        fz_f, fz_r = m * G * LR / sim.WHEELBASE, m * G * LF / sim.WHEELBASE
        if self.realism >= 2:   # the servo turns the wheels toward the command at a limited rate
            self.wheel = self.wheel + np.clip(cmd[:, 0] * STEER_MAX - self.wheel, -SERVO_RATE * dt, SERVO_RATE * dt)
        else:
            self.wheel = cmd[:, 0] * STEER_MAX
        delta = self.wheel
        thr = cmd[:, 1]
        f_drive = np.where(thr >= 0, thr * np.minimum(MASS0 * sim.ACCEL, POWER / np.maximum(self.vx, 1.0)), 0.0)
        f_brake = np.where(thr < 0, -thr * MASS0 * sim.BRAKE, 0.0) * (self.vx > 0.05)
        fx_f = DRIVE_SPLIT[0] * f_drive - BRAKE_SPLIT[0] * f_brake
        fx_r = DRIVE_SPLIT[1] * f_drive - BRAKE_SPLIT[1] * f_brake
        pk_f, pk_r = mu * fz_f, mu * fz_r
        if self.realism >= 2:
            # asking for more than the peak locks (brake) or spins (drive) the wheel: it slides, and a sliding
            # tire grips less (SLIDE_FRAC of the peak) and has nothing left for turning
            self.locked = np.stack([np.abs(fx_f) > pk_f, np.abs(fx_r) > pk_r], 1)
            fx_f = np.where(self.locked[:, 0], np.sign(fx_f) * SLIDE_FRAC * pk_f, fx_f)
            fx_r = np.where(self.locked[:, 1], np.sign(fx_r) * SLIDE_FRAC * pk_r, fx_r)
        fx_f, fx_r = np.clip(fx_f, -pk_f, pk_f), np.clip(fx_r, -pk_r, pk_r)
        # friction ellipse: what longitudinal force uses up is not available sideways
        left_f = np.sqrt(np.maximum(1 - (fx_f / pk_f) ** 2, 0.0))
        left_r = np.sqrt(np.maximum(1 - (fx_r / pk_r) ** 2, 0.0))
        if self.realism >= 2:
            left_f, left_r = np.where(self.locked[:, 0], 0.0, left_f), np.where(self.locked[:, 1], 0.0, left_r)
        vx_s = np.maximum(self.vx, V_KIN)
        a_f = np.arctan2(self.vy + LF * self.yaw_rate, vx_s) - delta
        a_r = np.arctan2(self.vy - LR * self.yaw_rate, vx_s)
        fy_f = tire_force(a_f, pk_f) * left_f
        fy_r = tire_force(a_r, pk_r) * left_r
        if self.realism >= 2:   # relaxation: sideways force builds up over RELAX_LEN of rolling, not instantly
            k = 1 - np.exp(-np.maximum(self.vx, V_KIN) * dt / RELAX_LEN)
            self.fy = self.fy + k[:, None] * (np.stack([fy_f, fy_r], 1) - self.fy)
            fy_f, fy_r = self.fy[:, 0], self.fy[:, 1]
        cd, sd = np.cos(delta), np.sin(delta)
        self._acc_sum += np.stack([fx_f * cd - fy_f * sd + fx_r, fx_f * sd + fy_f * cd + fy_r], 1) / m[:, None]
        ax = (fx_f * cd - fy_f * sd + fx_r) / m + self.vy * self.yaw_rate
        ay = (fx_f * sd + fy_f * cd + fy_r) / m - self.vx * self.yaw_rate
        rdot = (LF * (fx_f * sd + fy_f * cd) - LR * fy_r) / iz
        vx = np.clip(self.vx + ax * dt, 0.0, sim.V_MAX)
        vy = self.vy + ay * dt
        r = self.yaw_rate + rdot * dt
        # low speed: blend toward the kinematic model (no slip: rear axle moves straight ahead)
        lam = np.clip((vx - V_KIN) / (V_DYN - V_KIN), 0, 1)
        r_kin = vx * np.tan(delta) / sim.WHEELBASE
        vy_kin = r_kin * LR
        r = lam * r + (1 - lam) * r_kin
        vy = lam * vy + (1 - lam) * vy_kin
        vmid_x, vmid_y = 0.5 * (self.vx + vx), 0.5 * (self.vy + vy)
        self.vx, self.vy, self.yaw_rate = vx, vy, r
        self.heading = self.heading + r * dt
        c, s = np.cos(self.heading), np.sin(self.heading)
        self.pos = self.pos + np.stack([c * vmid_x - s * vmid_y, s * vmid_x + c * vmid_y], 1) * dt
        self.slip = np.stack([a_f, a_r], 1)

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=float).reshape(self.n, 2), -1, 1)
        self.hist = np.roll(self.hist, 1, axis=1)
        self.hist[:, 0] = a
        dt = sim.DT / SUBSTEPS
        rows = np.arange(self.n)
        self._acc_sum = np.zeros((self.n, 2))
        for j in range(SUBSTEPS):
            # the command reaching the wheels at this instant was issued `delay` seconds ago
            lag = np.clip(np.ceil(np.round((self.delay - j * dt) / sim.DT, 9)), 0, MAX_DELAY_STEPS).astype(int)
            self._substep(self.hist[rows, lag], dt)
        self.accel = self._acc_sum / SUBSTEPS
        self.v = np.hypot(self.vx, self.vy)
        self.steer = self.wheel.copy()
        ds = self._locate()
        self.progress += ds
        self.steps += 1
        off = np.abs(self.lateral) > sim.OFFTRACK_LIMIT
        if self.reward_mode == "progress":
            r = ds / (sim.V_MAX * sim.DT)
        elif self.reward_mode == "survival":
            r = np.ones(self.n)
        else:
            r = 1 - np.minimum(np.abs(self.lateral) / sim.OFFTRACK_LIMIT, 1)
        r = np.where(off, -self.crash_penalty, r)
        self.ep_return += r
        lap_done = (self.progress >= self.length * self.laps) if self.laps else np.zeros(self.n, bool)
        terminated = off
        truncated = ~off & ((self.steps >= self.max_steps) | lap_done)

        self._cast_rays()
        self._make_obs()
        obs = self.obs()
        infos = [{} for _ in range(self.n)]
        done = terminated | truncated
        for i in np.flatnonzero(done):
            infos[i] = dict(
                terminal_observation=obs[i].copy(),
                episode=dict(r=float(self.ep_return[i]), l=int(self.steps[i])),
                laps=float(self.progress[i] / self.length[i]),
                crashed=bool(terminated[i]),
                track=int(self.seeds[i]),
                mean_speed=float(self.progress[i] / (self.steps[i] * sim.DT)),
                physics=dict(grip=float(self.grip[i]), mass=float(self.mass[i]), delay=float(self.delay[i]),
                             noise=float(self.noise[i])),
                final_state=dict(pos=self.pos[i].copy(), heading=float(self.heading[i]), v=float(self.v[i]),
                                 rays=self.rays[i].copy(), lateral=float(self.lateral[i]),
                                 slip=self.slip[i].copy(), yaw_rate=float(self.yaw_rate[i])),
                **{"TimeLimit.truncated": bool(truncated[i])},
            )
            self._reset_car(i)
        if done.any():
            self._cast_rays()
            fresh = self._obs.copy()
            self._make_obs()
            obs[done] = self._obs[done]
            fresh[done] = self._obs[done]
            self._obs = fresh
        self._push_history()
        return obs, r.astype(np.float32), terminated, truncated, infos


# ---- adapters ------------------------------------------------------------------------------

def phase1_policy(policy12):
    """Run a Phase 1 policy (12-number obs, speed-limited steering) unchanged in the dynamic car.

    It gets the first 12 observation numbers (rays + speed), and its steering, which meant
    'fraction of the steering that perfect grip allows at this speed', is converted to a
    wheel angle the same way Phase 1 did: steer × steer_limit(v) / STEER_MAX.
    """
    def act(b: DynamicRacerBatch):
        o = b.obs()[:, :sim.OBS_DIM]
        a = np.array(policy12(o), dtype=float).reshape(b.n, 2)
        a[:, 0] = np.clip(a[:, 0], -1, 1) * sim.steer_limit(b.vx) / STEER_MAX
        return a
    return act


class DynamicPursuit:
    """The hand-coded driver for the dynamic car. Privileged: it reads the track AND the grip.

    It knows the centerline, the episode's grip and mass, and where the dirt is (a driver
    who has walked the course), but not the delay or the sensor noise.
    Steering: pure pursuit wheel angle. Speed: at each point in the next PLAN_DIST metres
    the corner limit is sqrt(lat_margin·μ·g/κ) with that point's μ; it may go faster now
    only if it can brake down in time at brake_margin·μ·g. Brake command is capped at
    max_brake of what the tires hold here, so it never locks the wheels; throttle likewise,
    so it never spins them.
    """

    def __init__(self, lat_margin=0.7, brake_margin=0.5, k_lookahead=0.5, min_lookahead=4.0, gain=0.5,
                 max_brake=0.7, knows_grip=True):
        from sdc.racer.pursuit import PLAN_DIST
        self.plan = PLAN_DIST
        self.lat_margin, self.brake_margin = lat_margin, brake_margin
        self.k, self.l0, self.gain, self.max_brake = k_lookahead, min_lookahead, gain, max_brake
        self.knows_grip = knows_grip

    def __call__(self, b: DynamicRacerBatch) -> np.ndarray:
        N = trk.N_POINTS
        r = np.arange(b.n)
        look = self.l0 + self.k * b.v
        tgt = (b.idx + np.ceil(look / b.ds).astype(int)) % N
        to = b.center[r, tgt] - b.pos
        alpha = (np.arctan2(to[:, 1], to[:, 0]) - b.heading + np.pi) % (2 * np.pi) - np.pi
        delta = np.arctan2(2 * sim.WHEELBASE * np.sin(alpha), np.linalg.norm(to, axis=1))
        steer = np.clip(delta / STEER_MAX, -1, 1)
        n_ahead = int(np.ceil(self.plan / b.ds.min()))
        offs = np.arange(n_ahead)
        ahead = (b.idx[:, None] + offs[None]) % N
        if self.knows_grip:
            mu = GRIP0 * b.grip[:, None] * np.where(b.dirt_pts[r[:, None], ahead], DIRT_GRIP, 1.0)
        else:
            mu = np.full(ahead.shape, GRIP0)
        a_lim = mu * G
        kappa = np.abs(b.curv[r[:, None], ahead])
        v_corner = np.minimum(sim.V_MAX, np.sqrt(self.lat_margin * a_lim / np.maximum(kappa, 1e-6)))
        brake_a = self.brake_margin * np.minimum.accumulate(a_lim, axis=1)   # weakest grip on the way there
        v_allow = np.sqrt(v_corner ** 2 + 2 * brake_a * offs[None] * b.ds[:, None])
        max_brake = self.max_brake * a_lim[:, 0] / sim.BRAKE
        drive_max = self.max_brake * a_lim[:, 0] * b.mass / np.minimum(sim.ACCEL, POWER / (MASS0 * np.maximum(b.vx, 1.0)))
        throttle = np.clip(self.gain * (v_allow.min(axis=1) - b.v), -max_brake, np.minimum(drive_max, 1))
        return np.stack([steer, throttle], axis=1)
