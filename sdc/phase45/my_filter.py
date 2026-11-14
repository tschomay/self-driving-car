"""The owner's attitude estimator (learner-owned, Phase 4.5): a complementary filter for pitch and roll.

Owner's choices (2026-11-14):
  1. Blend alpha = 0.98 at 50 Hz: each step keeps 98% of the gyro-propagated angle and pulls 2% toward the
     accelerometer's angle, so the accelerometer corrects gyro drift within about 1 s.
  2. Ignore the accelerometer whenever the total acceleration isn't close to 1 g (in the air it reads about 0,
     landings spike it, hard braking tilts it). "Close" = within G_TOL of 1 g (agent's default; the owner may change it).
  3. Gyro bias: average the gyro while the car sits still at the start (the first 0.2 s) and subtract it.
  4. Both pitch and roll.
  5. (E3, owner: "yes") Telemetry: subtract the car's own forward acceleration, measured from the motor speed,
     from the accelerometer's forward reading before taking its angle. Motor speed comes in 0.14 m/s steps every
     20 ms, so the acceleration is measured over a 0.1 s window (ACC_WINDOW steps; agent's default).

Conventions (body frame: x forward, y left, z up; the IMU reports specific force, so at rest it reads +1 g on z):
  pitch > 0 = nose up; roll > 0 = left side up.
  nose-up pitch rate = -gyro_y; left-up roll rate = +gyro_x (small-angle: fine for the ±30° this car sees).
  accelerometer angles from the gravity direction: pitch = atan2(a_x, sqrt(a_y² + a_z²)), roll = atan2(a_y, a_z).
"""

from __future__ import annotations

import numpy as np

G = 9.81
DT = 0.02
ALPHA = 0.98        # owner, choice 1
G_TOL = 0.15        # g: accelerometer trusted only when |a| is within 1 ± 0.15 g (owner's choice 2; tolerance: agent)
ACC_WINDOW = 5      # steps: forward acceleration = motor-speed change over 0.1 s (owner's choice 5; window: agent)


def accel_angles(acc):
    """(pitch, roll) in rad from one accelerometer reading (m/s², body frame)."""
    ax, ay, az = acc
    return np.arctan2(ax, np.hypot(ay, az)), np.arctan2(ay, az)


class AttitudeFilter:
    def __init__(self, alpha=ALPHA, g_tol=G_TOL, dt=DT, telemetry=False, window=ACC_WINDOW):
        self.alpha, self.g_tol, self.dt = alpha, g_tol, dt
        self.telemetry, self.window = telemetry, window
        self.reset()

    def reset(self, still_gyro=None, still_acc=None):
        """Start of a run. still_gyro / still_acc: (n, 3) readings while the car sits still (the first 0.2 s)."""
        # owner, choice 3: the gyro's average while still is its bias
        self.bias = np.zeros(3) if still_gyro is None else np.mean(still_gyro, axis=0)
        # start from the accelerometer's angle while still (gravity is the only acceleration then)
        self.pitch, self.roll = (0.0, 0.0) if still_acc is None else accel_angles(np.mean(still_acc, axis=0))
        self.trusted = True
        self.speeds = []

    def forward_accel(self, speed):
        """The car's own forward acceleration (m/s²) from the motor speed history (owner's choice 5)."""
        self.speeds = (self.speeds + [float(speed)])[-(self.window + 1):]
        if len(self.speeds) <= self.window:
            return 0.0
        return (self.speeds[-1] - self.speeds[0]) / (self.window * self.dt)

    def update(self, acc, gyro, speed=None):
        """One control step: acc (m/s²), gyro (rad/s), body frame; speed: motor speed (m/s) for telemetry.
        Returns (pitch, roll) in rad."""
        acc = np.array(acc, float)
        # 2) owner, choice 2: only blend in the accelerometer when it feels about 1 g. Checked on the raw reading:
        # in the air the wheels spin up freely, and subtracting that "acceleration" could fake 1 g in free fall.
        self.trusted = abs(np.linalg.norm(acc) / G - 1.0) <= self.g_tol
        if self.telemetry and speed is not None:
            # owner, choice 5: the push along the car isn't tilt; take it out of the forward reading
            acc[0] -= self.forward_accel(speed)
        g = np.asarray(gyro) - self.bias
        # 1) propagate with the gyro (short-term accurate, drifts)
        pitch = self.pitch - g[1] * self.dt
        roll = self.roll + g[0] * self.dt
        if self.trusted:
            ap, ar = accel_angles(acc)
            # owner, choice 1: 98% gyro path, 2% accelerometer
            pitch = self.alpha * pitch + (1 - self.alpha) * ap
            roll = self.alpha * roll + (1 - self.alpha) * ar
        # owner, choice 4: both angles
        self.pitch, self.roll = float(pitch), float(roll)
        return self.pitch, self.roll
