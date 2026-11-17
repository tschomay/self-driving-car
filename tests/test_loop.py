"""Phase 4.6 loop courses: geometry, heightfield, sensors' ray march, and one lap with the teacher."""
import numpy as np

import sdc.jumps  # noqa: F401  (sets the GL backend before mujoco loads)
import mujoco

from sdc.jumps import course as crs
from sdc.jumps import driver as dr
from sdc.jumps import env as je
from sdc.jumps import loop as lp
from sdc.phase46.teacher import LoopExpert


def test_loop_is_deterministic_and_obeys_its_rules():
    a, b = lp.make(10003), lp.make(10003)
    assert np.array_equal(a.center, b.center) and a.to_dict() == b.to_dict()
    for seed in (10000, 10003, 46001):
        c = lp.make(seed)
        assert lp.LAP[0] <= c.length <= lp.LAP[1]
        assert np.abs(c.kappa).max() <= 1 / lp.MIN_RADIUS
        assert np.abs(c.center).max() < lp.BOX / 2 - lp.STRIP
        assert len(c.jumps) == lp.N_JUMPS and all(j.kind in lp.KINDS for j in c.jumps)
        for j in c.jumps:   # every jump sits on a straight, with room before the line
            i = np.arange(int(j.start / c.ds), int(j.end / c.ds))
            assert np.abs(c.kappa[i]).max() < lp.STRAIGHT and j.end < c.length - 6.0
        assert c.profile[:int(8 / c.ds)].max() == 0.0   # a flat run-up from the start line


def test_track_coordinates_round_trip():
    c = lp.make(10000)
    for s, d in ((5.0, 0.0), (40.0, 0.8), (c.length - 3.0, -1.1), (77.7, 1.4)):
        X, Y, th = c.pose(s)
        X, Y = X - d * np.sin(th), Y + d * np.cos(th)
        s2, d2, _, _ = c.project(X, Y)
        assert abs((s2 - s + c.length / 2) % c.length - c.length / 2) < 0.08 and abs(d2 - d) < 0.05


def test_heightfield_matches_the_centreline_and_has_banks():
    c = lp.make(10000)
    hf = c.hfield() * lp.ZMAX
    assert hf.min() > 0.05 and hf.max() < lp.ZMAX
    for s in np.linspace(0, c.length, 60, endpoint=False):
        X, Y, th = c.pose(s)
        z = hf[int(round((Y + lp.BOX / 2) / lp.D)), int(round((X + lp.BOX / 2) / lp.D))]
        assert abs(z - c.height(s)) < 0.03
        Xb, Yb = X - 1.95 * np.sin(th), Y + 1.95 * np.cos(th)       # on top of the left bank
        zb = hf[int(round((Yb + lp.BOX / 2) / lp.D)), int(round((Xb + lp.BOX / 2) / lp.D))]
        assert zb > c.height(s) + 0.1 or c.jump_at(s) is not None


def test_ray_march_matches_mujoco():
    car = je.JumpCar(np.random.default_rng(0), real="all", sensors="parts", loop=True)
    car.reset(10000)
    R = car.d.xmat[1].reshape(3, 3)
    o = car.d.xpos[1] + R @ np.array([0.235, 0, 0.02])
    dirs = car.tof_dirs @ R.T
    ref, gid = np.empty(64), np.empty(64, np.int32)
    mujoco.mj_multiRay(car.m, car.d, o, dirs.ravel(), np.array([1, 0, 0, 0, 0, 0], np.uint8), True, -1, gid, ref,
                       None, 64, 4.0)
    ref[(ref < 0) | (ref > 4.0)] = 4.0
    assert np.abs(lp.cast(car.course, o, dirs, 4.0) - ref).max() < 0.03


def test_teacher_drives_a_lap_and_state_is_in_track_coordinates():
    eps = dr.run(LoopExpert(), [10000], n_envs=1, real="all", sensors="parts", attitude=True, loop=True,
                 max_steps=3000)
    e = eps[0]
    assert e["finished"] and e["n_jumps"] == 3 and e["landed"] >= 2
    assert e["x"] >= lp.make(10000).length - 0.5


def test_straight_courses_are_unchanged_by_the_loop_option():
    a = je.JumpCar(np.random.default_rng(0))
    assert a.legacy and not a.loop and a.m.hfield_nrow[0] == crs.NROW
    b = je.JumpCar(np.random.default_rng(0), loop=True)
    assert b.m.hfield_nrow[0] == lp.N


def test_angled_sensors_append_two_inputs():
    a = je.JumpCar(np.random.default_rng(0), real="all", sensors="parts", attitude=True, loop=True)
    b = je.JumpCar(np.random.default_rng(0), real="all", sensors="parts", attitude=True, loop=True, angled=True)
    oa, ob = a.reset(10000), b.reset(10000)
    assert len(ob) == len(oa) + 2
    assert np.all(np.abs(ob[-2:] * 3.0 - 1.5 / np.cos(np.pi / 4)) < 0.4)   # centred on a 3 m lane: about 2.1 m each


def test_dead_reckoning_closes_a_square():
    from sdc.phase46 import my_deadreckoning as mdr
    est = mdr.DeadReckoning(start_x=0.0)
    est.reset(np.zeros((10, 3)))
    for leg in range(4):                     # 4 x (2 s straight at 2 m/s, then a 90° left turn over 1 s)
        for _ in range(100):
            est.update([0, 0, 0], 2.0)
        for _ in range(50):
            est.update([0, 0, np.pi / 2], 0.0)
    path, (err_xy, err_h) = est.at_line()
    assert err_xy < 0.2 and abs(err_h) < 1e-6 and abs(path[-1, 0] - 16.0) < 1e-6
    assert np.hypot(path[-1, 1], path[-1, 2]) < 1e-9          # "spread": the mapped loop closes
