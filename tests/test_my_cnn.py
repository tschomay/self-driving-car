"""Checks for the owner's CNN encoder (sdc/phase2/my_cnn.py). Skipped until the OWNER TODOs are filled in."""

import numpy as np
import pytest
import torch

from sdc.phase2 import my_cnn
from sdc.racer.camera_env import obs_space

pytestmark = pytest.mark.skipif(my_cnn.conv_layers(1) is None or my_cnn.FEATURES_DIM is None,
                                reason="OWNER TODOs in sdc/phase2/my_cnn.py not filled in yet")


@pytest.mark.parametrize("frames,speed", [(1, True), (4, False)])
def test_encoder_shapes(frames, speed):
    enc = my_cnn.MyEncoder(obs_space(frames, speed))
    obs = {"image": torch.rand(5, frames, 64, 64)}
    if speed:
        obs["speed"] = torch.rand(5, 1)
    out = enc(obs)
    assert out.shape == (5, my_cnn.FEATURES_DIM + int(speed))
    assert torch.isfinite(out).all()
    assert any(isinstance(m, torch.nn.Conv2d) for m in enc.convs)


def test_encoder_sees_the_image():
    enc = my_cnn.MyEncoder(obs_space(1, False))
    a = enc({"image": torch.zeros(1, 1, 64, 64)})
    b = enc({"image": torch.rand(1, 1, 64, 64)})
    assert not torch.allclose(a, b)
