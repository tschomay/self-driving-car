import numpy as np

from sdc.jumps import course as crs
from sdc.jumps import driver as dr
from sdc.jumps import env as je


def test_course_deterministic_and_fits():
    a, b = crs.make(10000), crs.make(10000)
    assert np.array_equal(a.profile, b.profile)
    assert len(a.jumps) == crs.N_JUMPS
    for s in list(range(200)) + crs.TEST_SEEDS:
        c = crs.make(s)
        assert c.jumps[-1].end < crs.FINISH_X - 8
        assert all(j.start < j.lip < j.land_start <= j.end for j in c.jumps)
    hf = a.hfield()
    assert hf.shape == (crs.NROW, crs.NCOL) and hf.min() >= 0 and hf.max() <= 1


def test_test_and_val_sets_disjoint():
    assert not set(crs.TEST_SEEDS) & set(crs.VAL_SEEDS)


def test_batch_shapes_and_reset():
    b = je.JumpBatch(2, seed=0)
    obs = b.reset()
    assert obs.shape == (2, je.OBS_DIM) and np.isfinite(obs).all()
    for _ in range(5):
        obs, rew, term, trunc, infos = b.step(np.zeros((2, 2)))
    assert obs.shape == (2, je.OBS_DIM) and rew.shape == (2,)


def test_resting_car_counts_as_on_the_ground():
    b = je.JumpBatch(1, courses=[10000])
    b.reset()
    c = b.cars[0]
    for _ in range(20):
        c.step(np.array([0.0, 0.0]))
        wheels, body = c.contacts()
        assert wheels.all() and not body


def test_stopped_car_gets_stuck_and_misses_every_jump():
    b = je.JumpBatch(1, courses=[10000])
    b.reset()
    for _ in range(200):
        _, _, term, trunc, infos = b.step(np.zeros((1, 2)))
        if term[0]:
            break
    ep = infos[0]["episode"]
    assert ep["reason"] == "stuck" and ep["landed"] == 0
    assert all(j.get("missed") for j in ep["jumps"])


def test_expert_lands_a_course():
    ep = dr.run(dr.ExpertDriver(), [10000])[0]
    assert ep["finished"] and ep["landed"] == ep["n_jumps"]
    assert all(j["air"] >= je.MIN_AIR for j in ep["jumps"])
