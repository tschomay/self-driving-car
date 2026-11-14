"""The 3D RC car in MuJoCo: a 1/10-scale 4WD off-road buggy on a heightfield jump course.

Car (body frame: x forward, y left, z up):
  - chassis 0.46 x 0.18 x 0.07 m (8 cm bumper overhang front and rear), 1.8 kg; four wheels (spheres, r = 5.5 cm, 0.12 kg) on sprung
    suspension (8 cm travel), front wheels steer up to ±0.4 rad through a position servo.
  - 4WD electric motor per wheel: torque = 0.25 N·m × throttle − back-EMF drag, top speed
    about 14 m/s. throttle < 0 brakes, then reverses.
  - The motor pushes against the chassis, so in the air spinning the wheels up tips the nose
    UP and braking them tips the nose DOWN: that's how a real RC car steers its pitch mid-air.

Physics: 2 ms steps (implicitfast); the policy acts at 50 Hz (10 physics steps per action).
The rays are cast once per action (mj_multiRay against the ground only), not as MuJoCo
rangefinder sensors, which would be recomputed every physics step at 10x the cost.

Sensors the policy may use (real sensors on a real car):
  accelerometer (3, body frame, includes gravity), gyro (3), wheel speeds (front, rear, m/s),
  8 range-finder rays from the front bumper (3° above to 50° below the chassis' forward axis,
  like a small lidar), 2 sideways rays (to the banks), and its own last action (2).
Oracle quantities (training/expert only, never the gate car): pose, velocity, terrain ahead.
"""

from __future__ import annotations

import numpy as np

import mujoco

from sdc.jumps import course as crs

TIMESTEP = 0.002
SUBSTEPS = 10                   # 50 Hz control
DT = TIMESTEP * SUBSTEPS
WHEEL_R = 0.055
TAU_MAX = 0.25                  # N·m per wheel
TOP_SPEED = 14.0                # m/s, back-EMF limit
STEER_MAX = 0.4                 # rad
RAY_ANGLES = (-3, 0, 2, 4, 7, 12, 25, 50)   # deg below the chassis' forward axis (-3 = slightly up)
RAY_CUTOFF = 6.0
WHEELS = ("fl", "fr", "rl", "rr")
CORNER = dict(fl=(0.15, 0.14), fr=(0.15, -0.14), rl=(-0.15, 0.14), rr=(-0.15, -0.14))
MOUNT_Z = -0.05


RAY_SITES = [f"ray{i}" for i in range(len(RAY_ANGLES))] + ["rayL", "rayR"]
# Optional horizontal fan (env fan=True): rays from the bumper at these angles left (+) / right (-) of
# straight ahead, level with the chassis, so the car sees the lane edges ahead and can tell its heading.
FAN_ANGLES = (20, 35, 60, -20, -35, -60)
FAN_SITES = [f"fan{i}" for i in range(len(FAN_ANGLES))]


def cast_rays(m, d, site_ids):
    """Distance along each ray site's z axis to the ground (geom group 0); RAY_CUTOFF if nothing within it."""
    origins = d.site_xpos[site_ids]
    vec = d.site_xmat[site_ids].reshape(len(site_ids), 3, 3)[:, :, 2]
    dist = np.empty(len(site_ids))
    for o in np.unique(origins, axis=0):   # mj_multiRay casts a fan from one point: one call per origin
        sel = np.flatnonzero((origins == o).all(1))
        sub, gid = np.empty(len(sel)), np.empty(len(sel), np.int32)
        mujoco.mj_multiRay(m, d, o.astype(np.float64), np.ascontiguousarray(vec[sel]).ravel(),
                           np.array([1, 0, 0, 0, 0, 0], np.uint8), True, -1, gid, sub, None, len(sel), RAY_CUTOFF)
        dist[sel] = sub
    dist[(dist < 0) | (dist > RAY_CUTOFF)] = RAY_CUTOFF
    return dist


def _site_zaxis(direction):
    d = np.asarray(direction, float)
    return " ".join(f"{v:.5f}" for v in d / np.linalg.norm(d))


# Phase 4.5 physics switches (model structure). Empty = Phase 4's car exactly.
#   tires: wide tires (a 3 cm tread with rounded shoulders: a capsule along the axle) instead of spheres
#   susp:  soft suspension (about 30% sag, about critically damped) and a higher centre of mass (battery on top)
#   drive: one motor through open differentials (equal torque to the wheels; back-EMF on the mean wheel speed)
MODEL_SWITCHES = ("tires", "susp", "drive")
TIRE_W = 0.03         # m, flat part of the tread (capsule segment)
SOFT_K = 185.0          # N/m per corner: 4.4 N static load per corner / 2.4 cm sag (30% of 8 cm travel)
SOFT_C = 18.0           # N·s/m per corner: about critically damped with the 0.45 kg corner load. At 0.4 the springs
                        # rebounded off the kicker lips (a pogo kick) and the expert landed 51%.
BATTERY_M, BATTERY_Z = 0.4, 0.05   # moved from the chassis box to a pack on top (centre of mass about 1 cm higher)


def model_xml(real=()) -> str:
    real = set(real)
    L, W = crs.LENGTH, crs.WIDTH
    rays = []
    for i, a in enumerate(RAY_ANGLES):
        r = np.deg2rad(a)
        rays.append(f'<site name="ray{i}" pos="0.235 0 0.02" zaxis="{_site_zaxis((np.cos(r), 0, -np.sin(r)))}" size=".005"/>')
    for i, a in enumerate(FAN_ANGLES):
        r, dn = np.deg2rad(a), 0.0
        v = (np.cos(r) * np.cos(dn), np.sin(r) * np.cos(dn), -np.sin(dn))
        rays.append(f'<site name="fan{i}" pos="0.235 {0.001 * (i + 1):.3f} 0.02" zaxis="{_site_zaxis(v)}" size=".005"/>')
    rays.append('<site name="rayL" pos="0 0.095 0.02" zaxis="0 1 0" size=".005"/>')
    rays.append('<site name="rayR" pos="0 -0.095 0.02" zaxis="0 -1 0" size=".005"/>')
    corners = []
    if "susp" in real:   # spring at rest at full droop (no preload): sags 2.4 cm under the static load
        susp = (f'<joint name="{{w}}_susp" type="slide" axis="0 0 1" range="-0.03 0.05" stiffness="{SOFT_K}" '
                f'springref="-0.03" damping="{SOFT_C}" limited="true" solreflimit="0.005 1" armature="0.01"/>')
    else:
        susp = ('<joint name="{w}_susp" type="slide" axis="0 0 1" range="-0.03 0.05" stiffness="1500" damping="30"\n'
                '               limited="true" solreflimit="0.005 1" armature="0.01"/>')
    if "tires" in real:  # a capsule along the axle: a 3 cm flat tread with fully rounded shoulders
        # (A true cylinder snags on the heightfield's grid in MuJoCo: up to 2.8 m/s lost in one landing step.)
        tire = (f'<geom name="{{w}}_tire" type="capsule" euler="1.5708 0 0" group="1" contype="1" conaffinity="0" '
                f'size="{WHEEL_R} {TIRE_W / 2}" mass="0.12" friction="1.0 0.005 0.0001" rgba=".15 .15 .15 1"/>')
    else:
        tire = (f'<geom name="{{w}}_tire" type="sphere" group="1" contype="1" conaffinity="0" size="{WHEEL_R}" mass="0.12" '
                f'friction="1.0 0.005 0.0001"\n                rgba=".15 .15 .15 1"/>')
    # back-EMF drag: per wheel (one motor each), or on the single motor's speed (the mean wheel speed, via a tendon)
    spin_damp = 1e-5 if "drive" in real else TAU_MAX / (TOP_SPEED / WHEEL_R)
    for w in WHEELS:
        cx, cy = CORNER[w]
        steer = (f'<joint name="{w}_steer" type="hinge" axis="0 0 1" range="-{STEER_MAX} {STEER_MAX}" '
                 f'damping="0.05" armature="0.0005"/>') if w[0] == "f" else ""
        corners.append(f'''
      <body name="{w}" pos="{cx} {cy} {MOUNT_Z}">
        <inertial pos="0 0 0" mass="0.05" diaginertia="2e-5 2e-5 2e-5"/>
        {susp.format(w=w)}
        {steer}
        <body name="{w}_wheel">
          <joint name="{w}_spin" type="hinge" axis="0 1 0" damping="{spin_damp:.6f}"/>
          {tire.format(w=w)}
        </body>
      </body>''')
    sensors = ['<accelerometer name="accel" site="imu"/>', '<gyro name="gyro" site="imu"/>']
    sensors += [f'<jointvel name="{w}_w" joint="{w}_spin"/>' for w in WHEELS]
    motors = "\n    ".join(f'<motor name="{w}_motor" joint="{w}_spin" gear="{TAU_MAX}" ctrlrange="-1 1"/>'
                           for w in WHEELS)
    tendon = ""
    if "drive" in real:   # one motor: equal torque to every wheel (open diffs), drag on the mean wheel speed
        motors = f'<motor name="motor" tendon="drive" gear="{4 * TAU_MAX}" ctrlrange="-1 1"/>'
        tendon = ('\n  <tendon><fixed name="drive" damping="' + f"{4 * TAU_MAX / (TOP_SPEED / WHEEL_R):.6f}" + '">'
                  + "".join(f'<joint joint="{w}_spin" coef="0.25"/>' for w in WHEELS) + '</fixed></tendon>')
    if "susp" in real:
        chassis_m, battery = 1.8 - BATTERY_M, (f'\n      <geom name="battery" type="box" group="1" contype="0" conaffinity="0" '
                                               f'pos="0 0 {BATTERY_Z}" size="0.07 0.035 0.015" mass="{BATTERY_M}" rgba=".2 .2 .2 1"/>')
    else:
        chassis_m, battery = 1.8, ""
    return f'''<mujoco model="rc-buggy">
  <compiler angle="radian"/>
  <option timestep="{TIMESTEP}" integrator="implicitfast"/>
  <visual><global offwidth="640" offheight="360"/><quality shadowsize="2048"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0 0 0"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.75 0.85 0.95" rgb2="0.95 0.97 1" width="64" height="64"/>
    <texture name="dirt" type="2d" builtin="checker" rgb1="0.62 0.50 0.34" rgb2="0.56 0.45 0.30" width="64" height="64"/>
    <material name="dirt" texture="dirt" texrepeat="60 4" reflectance="0"/>
    <hfield name="terrain" nrow="{crs.NROW}" ncol="{crs.NCOL}" size="{L / 2} {W / 2} {crs.ZMAX} 0.3"/>
  </asset>
  <worldbody>
    <geom name="ground" type="hfield" contype="0" conaffinity="1" hfield="terrain" pos="{L / 2} 0 0" friction="1.0 0.005 0.0001"
          material="dirt"/>
    <light name="sun" pos="30 -10 20" dir="-0.3 0.4 -1" directional="true" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <body name="car" pos="{crs.START_X} 0 0.25">
      <freejoint name="root"/>
      <camera name="chase" mode="trackcom" pos="-1.6 -1.6 0.9" xyaxes="0.707 -0.707 0 0.25 0.25 0.935" fovy="50"/>
      <geom name="chassis" type="box" group="1" contype="1" conaffinity="0" size="0.23 0.09 0.035" mass="{chassis_m}" rgba=".12 .44 .92 1"/>{battery}
      <site name="imu" pos="0 0 0" size=".01"/>
      {"".join(rays)}
      {"".join(corners)}
    </body>
  </worldbody>
  <actuator>
    {motors}
    <position name="fl_servo" joint="fl_steer" kp="3" ctrlrange="-{STEER_MAX} {STEER_MAX}"/>
    <position name="fr_servo" joint="fr_steer" kp="3" ctrlrange="-{STEER_MAX} {STEER_MAX}"/>
  </actuator>{tendon}
  <sensor>
    {chr(10).join(sensors)}
  </sensor>
</mujoco>'''


_BASE = {}


def build_model(real=()) -> mujoco.MjModel:
    """A fresh model (each env owns one so each can hold its own course in the heightfield)."""
    key = tuple(sorted(set(real) & set(MODEL_SWITCHES)))
    if key not in _BASE:
        _BASE[key] = model_xml(key)
    return mujoco.MjModel.from_xml_string(_BASE[key])
