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
