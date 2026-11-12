"""Plumbing test for the recurrent PPO, with a stand-in memory (torch's own GRU cell), so it runs
before and regardless of the owner's my_gru.py answers."""

import torch

from sdc.phase4 import my_gru
from sdc.phase4 import train_rnn


def _standin(monkeypatch, where="before", shared=True):
    cells = {}

    def gru_step(x, h, p):
        key = id(p)
        if key not in cells:
            cells[key] = torch.nn.GRUCell(x.shape[-1], h.shape[-1])
        return cells[key](x, h)

    monkeypatch.setattr(my_gru, "gru_step", gru_step)
    monkeypatch.setattr(my_gru, "reset_memory", lambda h, s: h * (1 - s[:, None]))
    monkeypatch.setattr(my_gru, "CONFIG", dict(hidden=8, where=where, shared=shared, seq_len=16))


def test_recurrent_training_runs(monkeypatch, tmp_path):
    _standin(monkeypatch, where="beside", shared=False)
    m = train_rnn.train(seed=0, total_timesteps=512, out=tmp_path / "r", eval_every=10**9, gif=False,
                        n_envs=2, n_steps=128, n_epochs=1)
    assert m["trained_timesteps"] >= 512
    pol, _ = train_rnn.load(tmp_path / "r")
    assert pol.net.cfg["hidden"] == 8


def test_memory_resets_at_episode_start(monkeypatch):
    _standin(monkeypatch)
    net = train_rnn.RecurrentActorCritic()
    obs = torch.zeros(2, 20)
    h = torch.ones(1, 2, 8)
    _, _, h1 = net.step(obs, h, torch.tensor([1.0, 0.0]))
    _, _, h0 = net.step(obs, torch.zeros(1, 2, 8), torch.tensor([0.0, 0.0]))
    assert torch.allclose(h1[:, 0], h0[:, 0]) and not torch.allclose(h1[:, 1], h0[:, 1])
