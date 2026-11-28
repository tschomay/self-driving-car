"""Behaviour tests of the training sim's truggy car on flat ground, for comparison with test/feel.mjs (same protocol).

    MUJOCO_GL=disable PYTHONPATH=<self-driving-car> python tools/py_feel.py [grip] [real]   (real: truggy2 by default)
"""
import json
import sys

import numpy as np

from sdc.jumps import car as cr
from sdc.jumps import env as je
from sdc.jumps import loop as lp

GRIP = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
REAL = sys.argv[2] if len(sys.argv) > 2 else "truggy2"
FLAT = 0.5   # m: ground level of the flattened heightfield


def make(nudge):
    b = je.JumpBatch(1, courses=[10001], real=REAL, sensors="ideal", loop=True, dr=None)
    c = b.cars[0]
    c.reset(10001)
    c.phys.friction = GRIP
    c._apply_phys()
    c.m.hfield_data[:] = FLAT / lp.ZMAX
    c.d.qpos[:] = c.m.qpos0
    c.d.qvel[:] = 0
    c.d.qpos[0:3] = [nudge, 0.0, FLAT + 0.17]
    c.d.qpos[3:7] = [1, 0, 0, 0]
    import mujoco
    mujoco.mj_forward(c.m, c.d)
    mujoco.mj_step(c.m, c.d, 100)
    return c


def st(c):
    d = c.d
    R = d.xmat[1].reshape(3, 3)
    v = d.qvel[:3]
    return dict(x=d.qpos[0], y=d.qpos[1], v=float(R[:, 0] @ v), yaw=float(np.arctan2(R[1, 0], R[0, 0])), yr=float(d.qvel[5]),
                sp=float(np.hypot(v[0], v[1])), pitch=float(np.arctan2(R[2, 0], np.hypot(R[0, 0], R[1, 0]))))


def air(nudge, u):
    """Free flight: level, 7 m/s, wheels spinning at ground speed, 2 m up; command u for 0.5 s. Pitch change, deg
    (nose up > 0, as env.JumpCar.state)."""
    import mujoco
    c = make(nudge)
    r = getattr(c, "wheel_r", cr.WHEEL_R)
    c.d.qpos[2] = FLAT + 2.0
    c.d.qvel[:] = 0
    c.d.qvel[0] = 7.0
    c.d.qvel[c.spin_dof] = 7.0 / r
    c.last_act = np.array([0, 0.5], np.float32)
    mujoco.mj_forward(c.m, c.d)
    p0 = st(c)["pitch"]
    tr = run(c, lambda t, s: [0, u], 0.5)
    return float(np.degrees(tr[-1]["pitch"] - p0))


def hold(v_t, v):
    return float(np.clip(0.8 * (v_t - v) + 0.06 * v_t, -1, 1))


def run(c, fn, secs):
    out = []
    for k in range(int(secs / cr.DT)):
        s = st(c)
        a = np.array(fn(k * cr.DT, s), np.float32)
        c._physics(a)
        c.t += 1
        c.last_act = a
        out.append(st(c))
    return out


def tests(nudge):
    r = {}
    tr = run(make(nudge), lambda t, s: [0, 1], 3.0)
    v = [s["v"] for s in tr]
    r["t6"] = next((k + 1) * cr.DT for k, x in enumerate(v) if x >= 6)
    r["t10"] = next(((k + 1) * cr.DT for k, x in enumerate(v) if x >= 10), None)
    r["v3"] = v[-1]
    c = make(nudge)
    run(c, lambda t, s: [0, hold(8, s["v"])], 3.0)
    s0 = st(c)
    tr = run(c, lambda t, s: [0, -1], 2.5)
    stop = next((k for k, s in enumerate(tr) if s["sp"] < 0.1), len(tr) - 1)
    r["brake_v0"] = s0["v"]
    r["brake_m"] = float(np.hypot(tr[stop]["x"] - s0["x"], tr[stop]["y"] - s0["y"]))
    r["brake_yaw"] = float(np.degrees(max(abs((s["yaw"] - s0["yaw"] + np.pi) % (2 * np.pi) - np.pi) for s in tr[:stop + 1])))
    for name, steer, vt in (("c5", 0.5, 5.0), ("c8", 0.5, 8.0)):
        tr = run(make(nudge), lambda t, s: [steer, hold(vt, s["v"])], 6.0)[-100:]
        yr, sp = np.mean([s["yr"] for s in tr]), np.mean([s["sp"] for s in tr])
        r[name + "_R"] = float(sp / abs(yr))
        r[name + "_g"] = float(sp * abs(yr) / 9.81)
    c = make(nudge)
    run(c, lambda t, s: [0, hold(7, s["v"])], 3.0)
    s0 = st(c)
    tr = run(c, lambda t, s: [1, 0], 2.0)
    r["lift_yaw"] = float(np.degrees(abs(tr[-1]["yaw"] - s0["yaw"] + np.pi) % (2 * np.pi) - np.pi)) if False else \
        float(np.degrees(np.sum(np.abs(np.diff(np.unwrap([s0["yaw"]] + [s["yaw"] for s in tr]))))))
    r["lift_v"] = tr[-1]["sp"]
    for name, u in (("air_thr", 1.0), ("air_brk", -1.0), ("air_neu", 0.0)):
        r[name] = air(nudge, u)
    return r


res = [tests(k * 1e-6) for k in range(5)]
print(json.dumps({k: [x[k] for x in res] for k in res[0]}))
