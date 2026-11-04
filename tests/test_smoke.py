"""Fast checks (~15 s): training, metrics, and report generation all run end to end."""

import json

from sdc import report
from sdc.phase0.train import random_baseline, train


def test_train_and_report(tmp_path):
    m = train("tuned", seed=0, total_timesteps=2_000, out=tmp_path / "ppo", eval_every=1_000,
              eval_episodes=2, final_eval_episodes=3, label="tiny")
    assert m["trained_timesteps"] >= 2_000
    assert m["episodes"] and m["evals"]
    assert (tmp_path / "ppo" / "rollout.gif").exists()
    r = random_baseline(tmp_path / "rand", n_episodes=5)
    assert 5 < r["final_eval"]["mean"] < 100
    saved = json.loads((tmp_path / "ppo" / "metrics.json").read_text())
    assert saved["label"] == "tiny"
    page = report.build([tmp_path / "rand", tmp_path / "ppo"], "Test Report", "q?", tmp_path / "r.html")
    text = page.read_text()
    assert "<svg" in text and "data:image/gif;base64," in text and "<title>Test Report</title>" in text


def test_overrides(tmp_path):
    m = train("default", seed=1, total_timesteps=512, out=tmp_path / "d", gif=False, eval_every=10_000,
              eval_episodes=1, final_eval_episodes=1, overrides={"n_envs": 2, "n_steps": 128, "learning_rate": 3e-3})
    assert m["n_envs"] == 2 and m["ppo"]["learning_rate"] == 3e-3
