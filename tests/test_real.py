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
