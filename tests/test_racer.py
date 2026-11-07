"""Fast checks for the Phase 1 racer: tracks, physics, rays, baseline, training plumbing, report."""

import json

import numpy as np

from sdc.racer import evaluate, pursuit, sim
from sdc.racer import track as trk


def test_tracks_deterministic_and_valid():
    a, b = trk.generate(3), trk.generate(3)
    assert np.allclose(a.center, b.center)
    assert not np.allclose(a.center, trk.generate(4).center)
    for s in (0, 1, 2, *trk.HELD_OUT_SEEDS[:3]):
        t = trk.generate(s)
        assert np.max(np.abs(t.curvature)) <= 1 / trk.MIN_RADIUS
        assert np.allclose(np.linalg.norm(np.diff(t.center, axis=0), axis=1), t.ds, rtol=0.05)


def test_rays_and_obs_on_straight_start():
    b = sim.RacerBatch(2, tracks=[0, 1], start="line")
    obs = b.reset()
    assert obs.shape == (2, sim.OBS_DIM) and obs.dtype == np.float32
    half = trk.WIDTH / 2
    # side rays (±90°) hit the edges about half a track width away; speed starts at 0
    assert np.allclose(b.rays[:, 0], half, atol=0.3) and np.allclose(b.rays[:, -1], half, atol=0.3)
    assert np.all(b.rays[:, sim.N_RAYS // 2] > 10)
    assert np.all(obs[:, -1] == 0)


def test_progress_reward_and_crash():
    b = sim.RacerBatch(1, tracks=[0], start="line")
    b.reset()
    total = 0.0
    for _ in range(20):  # accelerate straight on the start straight
        _, r, term, trunc, _ = b.step(np.array([[0.0, 1.0]]))
        total += r[0]
        assert not term[0]
    assert b.progress[0] > 10 and abs(total * sim.V_MAX * sim.DT - b.progress[0]) < 1e-6
    for _ in range(200):  # full left lock: must leave the road and terminate
        _, _, term, _, infos = b.step(np.array([[1.0, 1.0]]))
        if term[0]:
            assert infos[0]["crashed"] and "terminal_observation" in infos[0]
            break
    else:
        raise AssertionError("never crashed")


def test_pure_pursuit_finishes_heldout():
    r = evaluate.run_laps(pursuit.PurePursuit(), trk.HELD_OUT_SEEDS[:5])
    assert r["completion"] == 1.0 and 10 < r["mean_lap_time"] < 60
    assert evaluate.compare(r, r)["lap_time_ratio"] == 1.0


def test_train_and_report(tmp_path):
    from sdc.phase1 import report
    from sdc.phase1.train import pursuit_baseline, train

    pursuit_baseline(tmp_path / "pp", gif=False)
    m = train(seed=0, total_timesteps=1024, out=tmp_path / "ppo", eval_every=512, label="tiny",
              overrides={"n_envs": 2, "n_steps": 256, "batch_size": 128}, env={"tracks": [0, 1], "max_steps": 200})
    assert m["trained_timesteps"] >= 1024 and m["evals"] and m["gifs"]
    assert (tmp_path / "ppo" / m["gifs"][0]["file"]).exists()
    assert json.loads((tmp_path / "ppo" / "metrics.json").read_text())["label"] == "tiny"
    page = report.build([tmp_path / "pp", tmp_path / "ppo"], "Test Report", "q?", tmp_path / "r.html").read_text()
    assert "<svg" in page and "data:image/gif;base64," in page and "<title>Test Report</title>" in page
