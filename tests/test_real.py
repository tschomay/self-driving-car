"""Phase 4.5 realism: switches off = Phase 4's car; realistic sensors and physics behave as modelled."""
import numpy as np

from sdc.jumps import car as cr
from sdc.jumps import driver as dr
from sdc.jumps import env as je
from sdc.jumps import real as rl


def test_no_switches_is_phase4_car():
    assert cr.model_xml() == cr.model_xml(())
    c = je.JumpCar(np.random.default_rng(0))
    assert c.legacy
    assert not je.JumpCar(np.random.default_rng(0), real=("servo",)).legacy


def test_every_switch_builds_and_drives():
    for sensors, dim in (("noisy", je.OBS_DIM + 7), ("parts", je.PARTS_DIM)):
        b = je.JumpBatch(2, seed=0, real="all", sensors=sensors, dr="gate", fan=True, servo=True)
        obs = b.reset()
        assert obs.shape == (2, dim) and np.isfinite(obs).all()
        for _ in range(30):
            obs, *_ = b.step(np.array([[0.1, 0.6], [-0.1, 0.4]]))
        assert np.isfinite(obs).all() and np.abs(obs).max() < 50


def test_physics_draw_depends_only_on_course_and_start():
    a = dr.run(dr.ExpertDriver(), [10001, 10002], n_envs=1, real="all", sensors="noisy", dr="train", max_steps=60)
    b = dr.run(dr.ExpertDriver(), [10001, 10002], n_envs=2, real="all", sensors="noisy", dr="train", max_steps=60)
    assert [e["phys"] for e in a] == [e["phys"] for e in b]
    assert a[0]["phys"] != a[1]["phys"]
    assert all(rl.RANGES["train"]["mass"][0] <= e["phys"]["mass"] <= rl.RANGES["train"]["mass"][1] for e in a)


def test_command_delay_and_servo_speed():
    c = je.JumpCar(np.random.default_rng(0), real=("delay", "servo"))
    c.reset(10000)
    c.phys.delay, c.phys.servo_s60 = 0.03, 0.12
    c.step(np.array([1.0, 0.0]))
    # 30 ms delay: nothing arrives in the first 20 ms step
    assert c.servo_set == 0.0
    c.step(np.array([1.0, 0.0]))
    c.step(np.array([1.0, 0.0]))
    # arrived 30 ms in; the servo then moves at 60°/0.12 s = 8.7 rad/s, so 30 ms later it's at about 0.26 rad
    assert 0.2 < c.servo_set < 0.3


def test_motor_speed_is_quantized_and_imu_clips():
    rng = np.random.default_rng(0)
    q = 1 / (rl.HALL_PER_M * 0.02)
    v = rl.motor_speed(rng, 3.33)
    assert abs(v / q - round(v / q)) < 1e-9 and abs(v - 3.33) <= q
    imu = rl.IMU(rng, rl.Phys())
    imu.sample(np.array([0, 0, 40 * 9.81]), np.zeros(3), 0.0, 0.002)
    assert imu.clipped
    acc, _ = imu.read()
    assert acc[2] < 16.5 * 9.81


def test_noisy_sensors_alone_keep_phase4_physics():
    """Review 2026-11-14: a float64 servo command made 'sensors only' runs drift from Phase 4's physics."""
    acts = [np.array([0.7 * np.sin(0.13 * t), 0.6], np.float32) for t in range(200)]
    paths = []
    for kw in ({}, dict(sensors="noisy")):
        c = je.JumpCar(np.random.default_rng(0), **kw)
        c.reset(10001)
        for a in acts:
            c.step(a)
        paths.append(c.d.qpos.copy())
    assert np.array_equal(paths[0], paths[1])


def test_lower_friction_lowers_contact_friction():
    """Review 2026-11-14: MuJoCo takes the larger geom friction, so the ground must be scaled too."""
    import mujoco
    mus = []
    for f in (0.5, 1.5):
        c = je.JumpCar(np.random.default_rng(0), real=("tires",))
        c.reset(10000)
        c.phys.friction = f
        c._apply_phys()
        mujoco.mj_step(c.m, c.d, 5)
        tire = set(c.tires.tolist())
        mus.append([c.d.contact.friction[i][0] for i in range(c.d.ncon)
                    if {int(c.d.contact.geom1[i]), int(c.d.contact.geom2[i])} & tire][0])
    assert np.allclose(mus, [0.5, 1.5])


def test_tof_updates_at_15_hz():
    c = je.JumpCar(np.random.default_rng(0), sensors="parts")
    c.reset(10000)
    frames, last = 0, c.tof.copy()
    for _ in range(150):   # 3 s
        c.step(np.array([0.0, 0.5]))
        if not np.array_equal(c.tof, last):
            frames += 1
            last = c.tof.copy()
    assert 43 <= frames <= 46


def test_owner_filter_conventions_and_bias():
    from sdc.phase45 import my_filter as mf
    # car pitched 10° nose up, at rest: gravity seen in the body frame
    th = np.deg2rad(10)
    acc = np.array([np.sin(th), 0, np.cos(th)]) * 9.81
    f = mf.AttitudeFilter()
    f.reset(still_gyro=np.tile([0.01, -0.02, 0.0], (10, 1)), still_acc=np.tile(acc, (10, 1)))
    for _ in range(100):   # the biased gyro mustn't drift it
        p, r = f.update(acc, np.array([0.01, -0.02, 0.0]))
    assert abs(np.degrees(p) - 10) < 0.1 and abs(r) < 1e-3
    # free fall: accelerometer reads ~0 and is ignored; a nose-up rate (-gyro_y) raises pitch
    p0 = f.pitch
    f.update(np.zeros(3), np.array([0.01, -0.02 - 1.0, 0.0]))
    assert not f.trusted and abs(f.pitch - (p0 + 0.02)) < 1e-9


def test_still_calibration_is_recorded():
    c = je.JumpCar(np.random.default_rng(0), real="all", sensors="noisy", dr="train")
    c.reset(10000)
    assert c.still_gyro.shape == (10, 3)
    assert np.abs(np.linalg.norm(c.still_acc, axis=1) / 9.81 - 1).max() < 0.2


def test_filter_telemetry_removes_launch_tilt():
    from sdc.phase45 import my_filter as mf
    level = np.array([0.0, 0.0, 9.81])
    for telem, ok in ((False, False), (True, True)):
        f = mf.AttitudeFilter(telemetry=telem)
        f.reset(np.zeros((10, 3)), np.tile(level, (10, 1)))
        v = 0.0
        for _ in range(100):   # level car accelerating at 4 m/s²: the accelerometer feels it as tilt
            v += 4 * mf.DT
            f.update(level + [4.0, 0, 0], np.zeros(3), speed=v)
        assert (abs(np.degrees(f.pitch)) < 2) == ok


def test_roller_courses_keep_tables_and_drops_in_place():
    from sdc.jumps import course as crs
    for seed in crs.VAL_SEEDS:
        a, b = crs.make(seed), crs.make(seed, kinds=crs.KINDS45)
        assert [j.kind.replace("gap", "double") for j in a.jumps] == [j.kind for j in b.jumps]
        for ja, jb in zip(a.jumps, b.jumps):
            assert ja.lip == jb.lip and ja.design_speed == jb.design_speed
        assert b.jumps[-1].end < crs.FINISH_X - 8
