"""Champion cars in models/: each loads, matches its stored sim's observation size, and drives one step."""

import json
from pathlib import Path

import numpy as np
import pytest

MODELS = sorted(p for p in Path("models").iterdir() if (p / "model.pt").exists())


@pytest.mark.parametrize("run", MODELS, ids=[p.name for p in MODELS])
def test_champion_loads_and_drives(run):
    from sdc.jumps import env as je
    from sdc.phase46.ensemble import Ensemble
    meta = json.loads((run / "metrics.json").read_text())
    assert "champion" in meta and "env46" in meta
    b = je.JumpBatch(1, courses=[10001], **meta["env46"])
    assert b.obs_dim == meta["obs_dim"]
    b._obs = np.zeros((1, b.obs_dim), np.float32)
    b._obs[0] = b.cars[0].reset(10001)
    pol = Ensemble((str(run),))
    pol.reset(1)
    a = np.asarray(pol(b))
    assert a.shape == (1, 2) and np.all(np.abs(a) <= 1.0001)
