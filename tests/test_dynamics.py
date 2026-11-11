"""Fast checks for the Phase 3 dynamic racer: grip limit, delay, noise, dirt, baseline, adapters."""

import numpy as np

from sdc.racer import dynamics as dy
from sdc.racer import sim


def _car(**physics):
    b = dy.DynamicRacerBatch(1, tracks=[10001], start="line", physics=physics or "nominal", dirt=False)
    b.reset()
    return b


def test_obs_shape_and_contents():
    b = _car()
    o = b.obs()
    assert o.shape == (1, dy.OBS_DIM) and o.dtype == np.float32
    o2, *_ = b.step(np.array([[0.3, 0.5]]))
    assert np.allclose(o2[0, sim.N_RAYS + 1:sim.N_RAYS + 3], [0.3, 0.5])   # last command is observed


def test_cornering_never_beats_grip():
    """Full lock at speed: sideways acceleration stays within grip·g (the car slides instead)."""
    for grip in (0.5, 1.0):
        b = _car(grip=grip)
        b.vx[:] = 20.0
        for _ in range(20):
            b._substep(np.array([[1.0, 0.0]]), 0.01)
        lat = abs(b.vx[0] * b.yaw_rate[0])   # steady-ish v·r ≈ sideways acceleration
        assert lat <= 1.15 * grip * dy.GRIP0 * dy.G
        assert abs(np.degrees(b.slip[0, 0])) > 10   # front tires well past their peak


def test_delay_holds_back_the_command():
    fast, slow = _car(), _car(delay=0.1)
    for b in (fast, slow):
        b.step(np.array([[0.0, 1.0]]))
    assert fast.vx[0] > 0.3 and slow.vx[0] == 0.0
    slow.step(np.array([[0.0, 1.0]]))
    assert slow.vx[0] > 0.3


def test_noise_only_touches_observations():
    a, b = _car(), _car(noise=0.05)
    assert np.allclose(a.rays, b.rays)
    assert not np.allclose(a.obs(), b.obs())


def test_dirt_is_fixed_per_track():
    m1, m2 = dy.dirt_mask(10000), dy.dirt_mask(10000)
    assert np.array_equal(m1, m2) and 0 < m1.mean() < 0.5
    b = dy.DynamicRacerBatch(1, tracks=[10000], start="line", dirt=True)
    b.reset()
    on = np.flatnonzero(m1)[0]
    b.idx[:] = on
    assert np.isclose(b.surface_grip[0], dy.GRIP0 * dy.DIRT_GRIP)


def test_physics_draws_inside_range():
    b = dy.DynamicRacerBatch(64, physics="gate")
    b.reset()
    for k in ("grip", "mass", "delay", "noise"):
        lo, hi = getattr(dy.GATE_DR, k)
        v = getattr(b, k)
        assert v.min() >= lo and v.max() <= hi and v.std() > 0


def test_hand_coded_driver_laps_and_phase1_adapter():
    from sdc.racer import evaluate
    import functools
    mk = functools.partial(dy.DynamicRacerBatch, physics="nominal", dirt=False)
    r = evaluate.run_laps(dy.DynamicPursuit(), [10001, 10002], make_batch=mk)
    assert r["completion"] == 1.0
    b = _car()
    a = dy.phase1_policy(lambda o: np.ones((len(o), 2)))(b)
    assert a.shape == (1, 2) and np.isclose(a[0, 0], 1.0)   # at standstill the full lock is allowed


def test_extra_sensors_read_sideways_motion():
    b = dy.DynamicRacerBatch(1, tracks=[10001], start="line", dirt=False, sensors=("velocity", "accel"))
    o = b.reset()
    assert o.shape == (1, dy.OBS_DIM + 3)
    b.vx[:] = 15.0
    for _ in range(5):
        o, *_ = b.step(np.array([[0.5, 0.0]]))   # steady left turn: slides outward (−y), pulled inward (+y)
    vy, ax, ay = o[0, dy.OBS_DIM:]
    assert vy < 0 and ay > 0.2
    assert np.isclose(ay * dy.G, b.vx[0] * b.yaw_rate[0], rtol=0.3)   # sideways g ≈ v·(turn rate)


def test_history_remembers_recent_own_motion():
    b = dy.DynamicRacerBatch(2, tracks=[10001, 10002], start="line", dirt=False, sensors=("accel",), history=3)
    o = b.reset()
    p = len(b.proprio)   # speed, 2 commands, gyro, 2 accel
    assert p == 6 and o.shape == (2, dy.OBS_DIM + 2 + 2 * p)
    assert np.all(o[:, -2 * p:] == 0)                       # nothing remembered yet
    o1, *_ = b.step(np.array([[0.2, 1.0], [0.0, 1.0]]))
    o2, *_ = b.step(np.array([[0.4, 0.5], [0.0, 1.0]]))
    base = dy.OBS_DIM + 2
    assert np.allclose(o2[:, base:base + p], o1[:, b.proprio])            # 1 step ago
    assert np.allclose(o2[0, base + 1:base + 3], [0.2, 1.0])              # ...including the command then


def test_oracle_grip_sensor():
    b = dy.DynamicRacerBatch(1, tracks=[10000], start="line", dirt=True, physics=dict(grip=0.6), sensors=("grip",))
    o = b.reset()
    assert o.shape == (1, dy.OBS_DIM + 1)
    expect = 0.6 * (dy.DIRT_GRIP if dy.dirt_mask(10000)[b.idx[0]] else 1.0)
    assert np.isclose(o[0, -1], expect)


def _car3(**physics):
    b = dy.DynamicRacerBatch(1, tracks=[10001], start="line", physics=physics or "nominal", dirt=False, realism=3)
    b.reset()
    return b


def _run3(b, cmd, n):
    for _ in range(n):
        b._substep3(np.array([cmd], float), 0.01)


def test_realism3_fixes_review_findings():
    b = _car3()
    b.vx[:], b.vy[:], b.yaw_rate[:] = 0.5, 8.0, 2.0
    _run3(b, [0, 0], 1)
    assert b.vy[0] > 7 and b.yaw_rate[0] > 1.5           # a low-speed slide isn't wiped out
    b = _car3()
    b.vx[:] = 15.0
    b.wheel[:] = dy.STEER_MAX
    _run3(b, [1, -1], 30)
    assert b.yaw_rate[0] > -0.05                          # locked fronts no longer turn the car the wrong way
    b = _car3(grip=dy.DIRT_GRIP)
    b.vx[:] = 8.0
    b.wheel[:] = 0.25 * dy.STEER_MAX
    _run3(b, [0.25, 1.0], 30)
    assert b.locked[0].all() and b.yaw_rate[0] > 0.15     # spinning wheels still corner
    b = _car3()
    b.vx[:] = 20.0
    _run3(b, [0, 0], 100)
    assert 19.0 < b.vx[0] < 19.95                         # drag and rolling resistance slow a coasting car


def test_lose_the_lap_and_lap_bonus():
    b = dy.DynamicRacerBatch(1, tracks=[10001], start="line", dirt=False, realism=3, crash_cost="lap", lap_bonus=50.0)
    b.reset()
    b.lap_reward[:] = 30.0
    b.pos = b.pos + 5.0 * b.normal[0, 0]          # put the car off the road
    _, r, term, _, _ = b.step(np.zeros((1, 2)))
    assert term[0] and np.isclose(r[0], -30.0)     # the lap's progress reward is taken back
    b.reset()
    b.progress[:] = b.length[0] - 0.01             # about to complete a lap's distance
    b.vx[:] = 10.0
    _, r, _, _, _ = b.step(np.array([[0.0, 0.5]]))
    assert r[0] > 50.0 and b.lap_reward[0] == 0.0


def test_gripmap_sensor_reads_dirt_ahead():
    import numpy as np
    from sdc.racer import dynamics as dy
    b = dy.DynamicRacerBatch(2, sensors=("accel", "gripmap"), realism=3, history=3, seed=0)
    o = b.reset()
    assert o.shape[1] == b.obs_dim == dy.OBS_DIM + 2 + 8 + 2 * (len(b.proprio))
    gm = o[:, dy.OBS_DIM + 2:dy.OBS_DIM + 10]
    assert np.allclose(gm, b.grip_ahead(dy.GRIPMAP_AHEAD))
    assert set(np.round(np.unique(gm / b.grip[:, None]), 3)) <= {1.0, round(dy.DIRT_GRIP, 3)}
    # the map isn't in the remembered own-motion readings
    assert not set(range(dy.OBS_DIM + 2, dy.OBS_DIM + 10)) & set(b.proprio.tolist())
