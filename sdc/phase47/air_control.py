"""Phase 4.7 side study: how much does throttle or brake rotate the truggy in the air?

    python -m sdc.phase47.air_control            -> prints tables (docs/phases/air-control.md)

Free flight, no ground in reach: the "truggy" car (nominal physics) is lifted 5 m above the lane, level, at a takeoff
speed v with its wheels rolling at v, and one throttle command is held for 1 s through the env's own actuator path
(JumpCar._physics: battery, ESC brake, single motor through the diffs). Reports pitch and wheel surface speed over time.

It also checks the angular momentum budget: chassis pitch rate change against the wheels' spin change, with and
without counting the spin joints' armature, and runs what-if variants that edit a private copy of the model in place
(the sim and its defaults are untouched).
"""

from __future__ import annotations

import numpy as np

import mujoco

from sdc.jumps import car as cr
from sdc.jumps import env as je

T_FLIGHT = 1.0
MARKS = (0.25, 0.5, 0.6, 1.0)          # s: report pitch at these times (median air time in logs: 0.58 s)
SPEEDS = (6.0, 7.0, 8.0)
COMMANDS = {"brake": -1.0, "neutral": 0.0, "full": 1.0}


def make_car(real="truggy"):
    car = je.JumpCar(np.random.default_rng(0), courses=[0], real=real)
    car.reset(course_seed=0)
    return car


def inertia_budget(car):
    """(sum of wheel spin inertia from the bodies, sum of spin armature, whole car's pitch inertia about its COM)."""
    m, d = car.m, car.d
    root = m.body("car").id
    sub = [b for b in range(1, m.nbody) if m.body_rootid[b] == root]
    com = d.subtree_com[root]
    I = np.zeros((3, 3))
    for b in sub:
        R = d.ximat[b].reshape(3, 3)
        Ib = R @ np.diag(m.body_inertia[b]) @ R.T
        r = d.xipos[b] - com
        I += Ib + m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
    Rc = d.xmat[root].reshape(3, 3)
    I_pitch = float(Rc[:, 1] @ I @ Rc[:, 1])
    Iw = 0.0
    for w in cr.WHEELS:
        b = m.body(f"{w}_wheel").id
        R = d.ximat[b].reshape(3, 3)
        Iw += float(Rc[:, 1] @ (R @ np.diag(m.body_inertia[b]) @ R.T) @ Rc[:, 1])
    arm = float(m.dof_armature[car.spin_dof].sum())
    return Iw, arm, I_pitch, float(m.body_subtreemass[root])


def launch(car, v, z=20.0):
    """Lift the settled car z m up, level, moving forward at v with the wheels rolling at v."""
    d = car.d
    d.qpos[2] += z
    d.qpos[3:7] = 1.0, 0.0, 0.0, 0.0
    d.qvel[:] = 0.0
    d.qvel[0] = v
    d.qvel[car.spin_dof] = v / cr.WHEEL_R
    car.queue, car.applied = [], np.zeros(2, np.float32)
    car.airborne = True
    mujoco.mj_forward(car.m, d)


def fly(car, v, thr, T=T_FLIGHT, pulse=None):
    """Hold throttle thr (or thr for `pulse` s, then neutral) for T s. Returns arrays t, pitch change deg (integrated
    from the pitch rate, nose up > 0), pitch rate deg/s, mean wheel surface speed m/s."""
    launch(car, v)
    n = int(round(T / cr.DT))
    t, p, q, w = [0.0], [0.0], [0.0], [v]
    for k in range(n):
        u = thr if pulse is None or k * cr.DT < pulse else 0.0
        car.queue = [(car.sub, np.array([0.0, u], np.float32))]   # no command latency: the command acts at once
        car._physics(np.array([0.0, u], np.float32))
        t.append((k + 1) * cr.DT)
        q.append(float(np.degrees(-car.d.qvel[4])))
        p.append(p[-1] + 0.5 * (q[-1] + q[-2]) * cr.DT)   # integrated pitch rate: no wrap past +-90 deg
        w.append(car._mean_wheel_speed())
    return np.array(t), np.array(p), np.array(q), np.array(w)


def at(t, x, s):
    return float(np.interp(s, t, x))


def t90(t, w):
    """Time for the wheels to cover 90% of their total speed change over the flight."""
    dw = w - w[0]
    if abs(dw[-1]) < 1e-6:
        return float("nan")
    return float(t[np.argmax(dw / dw[-1] >= 0.9)])


def table(car, label):
    print(f"\n== {label} ==")
    print(f"{'v':>4} {'cmd':>8} | " + " ".join(f"pitch@{m:.2f}" for m in MARKS)
          + " | rate@0.6 peak_rate | wheel v0 -> v1.0  t90")
    out = {}
    for v in SPEEDS:
        for name, thr in COMMANDS.items():
            car.reset(course_seed=0)
            t, p, q, w = fly(car, v, thr)
            p = p - p[0]
            pk = q[np.argmax(np.abs(q))]
            out[(v, name)] = (t, p, q, w)
            print(f"{v:4.1f} {name:>8} | " + " ".join(f"{at(t, p, m):+10.1f}" for m in MARKS)
                  + f" | {at(t, q, 0.6):+8.0f} {pk:+9.0f} | {w[0]:5.2f} -> {w[-1]:6.2f}  {t90(t, w):.2f}")
    return out


def budget_check(car, runs):
    Iw, arm, Ip, M = inertia_budget(car)
    print(f"\nmass {M:.2f} kg; wheel spin inertia (bodies) 4 x {Iw / 4:.2e}; armature 4 x {arm / 4:.2e}; "
          f"pitch inertia {Ip:.4f} kg m^2")
    print(f"ratio bodies only {Iw / Ip:.4f}; with armature {(Iw + arm) / Ip:.4f}")
    print("momentum check (chassis rate change / wheel spin change) at 0.6 s:")
    for (v, name), (t, p, q, w) in runs.items():
        if v != 7.0 or name == "neutral":
            continue
        dws = (at(t, w, 0.6) - w[0]) / cr.WHEEL_R          # rad/s
        print(f"  {name:>7}: measured {np.radians(at(t, q, 0.6)) / dws:+.4f}  "
              f"(predicted: bodies only {Iw / Ip:+.4f}, with armature {(Iw + arm) / Ip:+.4f})")
    return Iw, arm, Ip


def what_if(label, edit, real="truggy"):
    """Re-run the 7 m/s case on a private model edited in place (edit(car) after every reset)."""
    car = make_car(real)
    rows = []
    for name, thr in COMMANDS.items():
        car.reset(course_seed=0)
        edit(car)
        t, p, q, w = fly(car, 7.0, thr)
        p = p - p[0]
        rows.append(f"{name}: {at(t, p, 0.6):+5.0f} deg @0.6 s ({at(t, p, 1.0):+5.0f} @1 s), {at(t, q, 0.6):+4.0f} deg/s, t90 {t90(t, w):.2f}")
    print(f"  {label:<46} " + " | ".join(rows))


def main():
    np.set_printoptions(precision=3, suppress=True)
    car = make_car("truggy")
    print(f"truggy switches: {sorted(car.real)}; motor {cr.motor(car.real)}; brake cap {je.BRAKE_CAP}; "
          f"nominal charge {car.phys.charge}, sag {car.phys.sag}")
    print(f"actuator gear {car.m.actuator_gear[:car.n_motor, 0]}, spin damping {car.m.dof_damping[car.spin_dof]}, "
          f"tendon damping {car.m.tendon_damping}")
    runs = table(car, "truggy, level launch, command held 1 s")
    Iw, arm, Ip = budget_check(car, runs)

    print("\n== pulses at 7 m/s (pulse, then neutral), pitch at 0.6 s ==")
    for thr in (1.0, -1.0):
        for pl in (0.1, 0.2):
            car.reset(course_seed=0)
            t, p, q, w = fly(car, 7.0, thr, pulse=pl)
            print(f"  {thr:+.0f} for {pl:.1f} s: {at(t, p - p[0], 0.6):+6.1f} deg, rate after {at(t, q, 0.6):+5.0f} deg/s")

    print("\n== what-if (private model copies, 7 m/s; brake / neutral / full) ==")
    sd = car.spin_dof
    bodies = [car.m.body(f"{w}_wheel").id for w in cr.WHEELS]

    def nothing(c):
        pass

    def no_arm(c):
        c.m.dof_armature[sd] = 0.0

    def arm_to_body(c, k=1.0):
        # move the armature into the wheel bodies' own spin inertia (axis = body y; the capsule's local frame
        # differs, so scale every principal moment of the wheel body to add `arm` about the axle)
        for b in bodies:
            c.m.body_inertia[b] *= 1.0 + k * cr.WHEEL_ARMATURE / _axle_inertia(c, b)
        c.m.dof_armature[sd] = 0.0

    def set_axle(c, target):
        for b in bodies:
            c.m.body_inertia[b] *= target / _axle_inertia(c, b)

    def brake_up(c):
        je.BRAKE_CAP = 1.0

    def scale_wheel(c, k):
        for b in bodies:
            c.m.body_inertia[b] *= k

    try:
        _what_ifs(nothing, no_arm, arm_to_body, set_axle, brake_up, scale_wheel)
    finally:
        je.BRAKE_CAP = 0.65


def _what_ifs(nothing, no_arm, arm_to_body, set_axle, brake_up, scale_wheel):
    what_if("as is (truggy)", nothing)
    what_if("no armature", no_arm)
    what_if("armature moved into the wheel bodies", arm_to_body)
    what_if("proposal: wheel axle inertia 3.4e-4, armature kept", lambda c: set_axle(c, 3.4e-4))
    what_if("proposal + brake cap 1.0", lambda c: (set_axle(c, 3.4e-4), brake_up(c)))
    what_if("wheel body inertia x3, no armature", lambda c: (scale_wheel(c, 3.0), no_arm(c)))
    what_if("wheel body inertia x3, armature kept", lambda c: scale_wheel(c, 3.0))
    what_if("wheel body inertia x4, armature kept", lambda c: scale_wheel(c, 4.0))



def _axle_inertia(car, b):
    Rc = car.d.xmat[car.m.body("car").id].reshape(3, 3)
    R = car.d.ximat[b].reshape(3, 3)
    return float(Rc[:, 1] @ (R @ np.diag(car.m.body_inertia[b]) @ R.T) @ Rc[:, 1])


if __name__ == "__main__":
    main()
