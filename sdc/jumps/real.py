"""Phase 4.5 realism: physics randomization, actuator models, and sensors modelled on real hobby parts.

Everything here is switched on per car (env.JumpCar(real=..., sensors=..., dr=...)); with the defaults the
car is Phase 4's exactly.

Physics switches (`real`, a set of names; "all" = every one):
  tires, susp, drive   model structure (car.MODEL_SWITCHES): wide tires (capsules), soft suspension + higher centre
                       of mass, one motor through open differentials
  servo                the steering servo is speed-limited (s per 60°) with a deadband, not an instant target
  battery              motor power scales with the pack's charge and sags under load
  delay                the command reaches the servo/ESC `delay` seconds after the sensors were read
                       (Pi compute + PWM frame); held in a queue at the 2 ms physics rate

Sensor modes (`sensors`):
  ideal    Phase 4's perfect sensors (and observation layout)
  noisy    Phase 4's layout, so a Phase 4 policy can drive unchanged, but every signal is realistic:
           IMU noise/bias/vibration/clipping/mount tilt, one motor-speed signal (quantized) in both
           wheel-speed slots, range noise and dropouts on every ray, and the servo slot reports the
           command (a hobby servo has no position feedback)
  parts    the parts list's layout (PARTS_DIM numbers): IMU (6), motor speed (1), TFmini-S (1),
           VL53L5CX 8x8 zones (64, updated at 15 Hz), two side VL53L1X (2), last action (2)

Randomization (`dr`): None/"nominal" (fixed nominal values), "train", or "gate" (wider than train, as in
Phase 3). Values are drawn per episode from a generator seeded by (course, start offset, phys_seed), so an
evaluation is the same however it is batched.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

G = 9.81
PHYS_SWITCHES = ("tires", "susp", "drive", "servo", "battery", "delay")


def switches(real):
    if real in ("all", ("all",), ["all"]):
        return frozenset(PHYS_SWITCHES)
    real = frozenset([real] if isinstance(real, str) else (real or ()))
    bad = real - set(PHYS_SWITCHES)
    assert not bad, f"unknown physics switches {bad}"
    return real


@dataclass
class Phys:
    """One draw of the car's physical parameters (nominal values as defaults)."""
    mass: float = 1.0          # x all body masses
    friction: float = 1.0      # tire-ground sliding friction
    spring: float = 1.0        # x suspension stiffness
    damping: float = 1.0       # x suspension damping
    motor: float = 1.0         # x motor torque
    charge: float = 0.95       # battery charge (x motor voltage: torque and top speed)
    sag: float = 0.07          # voltage lost at full current (fraction)
    servo_s60: float = 0.12    # servo speed, s per 60°
    deadband: float = 0.01     # rad: the servo ignores command changes smaller than this
    delay: float = 0.015       # s from sensor read to the command reaching the actuators
    tilt_p: float = 0.0        # deg, IMU mounted pitched by this much
    tilt_r: float = 0.0        # deg, IMU mounted rolled by this much
    noise: float = 1.0         # x every sensor noise level
    tof_range: float = 3.0     # m, VL53L5CX usable range today (sunlight shortens it)


# (lo, hi) ranges; a missing field stays nominal.
RANGES = dict(
    nominal={},
    train=dict(mass=(0.85, 1.15), friction=(0.7, 1.1), spring=(0.8, 1.25), damping=(0.7, 1.4), motor=(0.85, 1.1),
               charge=(0.85, 1.0), servo_s60=(0.08, 0.16), delay=(0.005, 0.03), tilt_p=(-1, 1), tilt_r=(-1, 1),
               noise=(0.5, 1.5), tof_range=(2.0, 4.0)),
    gate=dict(mass=(0.75, 1.3), friction=(0.55, 1.25), spring=(0.7, 1.4), damping=(0.6, 1.7), motor=(0.8, 1.15),
              charge=(0.8, 1.0), servo_s60=(0.06, 0.2), delay=(0.0, 0.045), tilt_p=(-2, 2), tilt_r=(-2, 2),
              noise=(0.5, 2.0), tof_range=(1.5, 4.0)),
)


def draw(dr, rng) -> Phys:
    rng_ = RANGES["nominal" if dr is None else dr]
    p = Phys()
    for f in fields(Phys):   # always draw every field, so the stream doesn't depend on the range
        u = rng.random()
        if f.name in rng_:
            lo, hi = rng_[f.name]
            setattr(p, f.name, float(lo + u * (hi - lo)))
    return p


# ---- sensors -------------------------------------------------------------------------------------------
# ICM-42688-class IMU, read by the microcontroller at 500 Hz (every physics step) and averaged per control step.
ACC_CLIP = 16 * G              # ±16 g full scale: hard landings clip
GYRO_CLIP = np.deg2rad(2000)   # ±2000 °/s
ACC_NOISE = 0.06               # m/s² per sample (sensor)
GYRO_NOISE = 0.004             # rad/s per sample
VIB_ACC = 2.0                  # m/s² per sample of motor/drivetrain vibration at top speed (rises with motor speed)
VIB_GYRO = 0.05                # rad/s per sample at top speed
ACC_BIAS = 0.1                 # m/s² per axis (per episode, 1 sigma)
GYRO_BIAS = np.deg2rad(0.5)    # rad/s per axis (per episode, 1 sigma)
GYRO_WALK = np.deg2rad(0.05)   # rad/s per sqrt(s): bias drift
# Motor speed from a sensored ESC: hall-sensor events per metre travelled (4-pole motor, 12 events per
# motor rev, about 10:1 gearing, 5.5 cm wheels). Counted per 20 ms control step.
HALL_PER_M = 12 * 10 / (2 * np.pi * 0.055)
# Range sensors
TFMINI_MAX = 8.0               # m, usable outdoors (12 m indoors)
TOF_HZ = 15.0                  # VL53L5CX in 8x8 mode
TOF_FOV = 45.0                 # deg, square
TOF_PITCH = 25.0               # deg, mounted looking this far down
SIDE_MAX = 3.0                 # m, VL53L1X side sensors outdoors
RANGE_NOISE = (0.02, 0.01)     # metres + fraction of the reading (1 sigma)
DROPOUT = 0.02                 # chance a beam returns nothing (reads as max range)
TOF_DROPOUT = 0.04             # per zone per frame (zone status invalid)


def tof_dirs():
    """(64, 3) unit vectors in the body frame for the 8x8 zones (row 0 = top), centred on TOF_PITCH down."""
    a = np.deg2rad((np.arange(8) - 3.5) * TOF_FOV / 8)
    out = []
    for el in a:                       # el > 0 = below the centre line
        for az in a[::-1]:             # left to right
            v = np.array([1.0, np.tan(az), -np.tan(el)])
            v /= np.linalg.norm(v)
            p = np.deg2rad(TOF_PITCH)   # pitch the sensor down
            R = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
            out.append(R @ v)
    return np.array(out)


class IMU:
    """ICM-42688-class IMU sampled every 2 ms physics step and averaged per control step.

    Each sample is clipped at full scale (so a landing spike saturates), then the average gets the sensor
    noise and vibration of n averaged samples (sigma / sqrt(n), drawn once per read for speed), the
    per-episode bias, the bias drift, and the mount tilt."""

    def __init__(self, rng, phys: Phys):
        self.rng, self.k = rng, phys.noise
        self.acc_bias = rng.normal(0, ACC_BIAS * self.k, 3)
        self.gyro_bias = rng.normal(0, GYRO_BIAS * self.k, 3)
        tp, tr = np.deg2rad(phys.tilt_p), np.deg2rad(phys.tilt_r)
        Ry = np.array([[np.cos(tp), 0, np.sin(tp)], [0, 1, 0], [-np.sin(tp), 0, np.cos(tp)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(tr), -np.sin(tr)], [0, np.sin(tr), np.cos(tr)]])
        self.mount = (Ry @ Rx).T    # body -> sensor
        self.clear()

    def clear(self):
        self.acc, self.gyro, self.n, self.frac, self.clipped = np.zeros(3), np.zeros(3), 0, 0.0, False

    def sample(self, acc, gyro, motor_frac, dt):
        a, g = self.mount @ acc, self.mount @ gyro
        self.clipped |= bool(np.any(np.abs(a) > ACC_CLIP))
        self.acc += np.clip(a, -ACC_CLIP, ACC_CLIP)
        self.gyro += np.clip(g, -GYRO_CLIP, GYRO_CLIP)
        self.frac += motor_frac
        self.n += 1
        self.dt = dt

    def read(self):
        n = max(self.n, 1)
        k, f, r = self.k, self.frac / n, np.sqrt(n)
        self.gyro_bias += self.rng.normal(0, GYRO_WALK * np.sqrt(n * getattr(self, "dt", 0.002)), 3)
        acc = self.acc / n + self.acc_bias + self.rng.normal(0, k * (ACC_NOISE + VIB_ACC * f) / r, 3)
        gyro = self.gyro / n + self.gyro_bias + self.rng.normal(0, k * (GYRO_NOISE + VIB_GYRO * f) / r, 3)
        self.clear()
        return acc, gyro


def motor_speed(rng, v_mean, prev_frac=0.0):
    """Motor speed as wheel-surface speed (m/s) from hall counts in one 20 ms window (quantized; a random
    phase of the first count)."""
    counts = np.floor(abs(v_mean) * HALL_PER_M * 0.02 + rng.random())
    return float(np.sign(v_mean) * counts / (HALL_PER_M * 0.02))


def ranges(rng, dist, max_range, k, dropout=DROPOUT):
    """Noisy readings of true distances (each beam: noise, dropouts, max range)."""
    d = np.asarray(dist, float)
    sig = k * (RANGE_NOISE[0] + RANGE_NOISE[1] * d)
    out = d + rng.normal(0, 1, d.shape) * sig
    out[(d >= max_range) | (rng.random(d.shape) < dropout)] = max_range
    return np.clip(out, 0.0, max_range)
