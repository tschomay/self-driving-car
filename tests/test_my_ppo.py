"""Known-answer checks for the owner's PPO pieces (sdc/phase1/my_ppo.py). Each skips until written."""

import pytest
import torch

from sdc.phase1 import my_ppo


def _call(fn, *args):
    try:
        return fn(*args)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not written yet")


def t(x):
    return torch.tensor(x, dtype=torch.float32)


def test_advantages_monte_carlo():
    # gamma = lam = 1, critic says 0 everywhere: advantage = sum of rewards still to come
    adv = _call(my_ppo.compute_advantages, t([[1.0], [1.0]]), t([[0.0], [0.0]]), t([[0.0], [0.0]]), t([0.0]), 1.0, 1.0)
    assert torch.allclose(torch.as_tensor(adv, dtype=torch.float32), t([[2.0], [1.0]]))


def test_advantages_episode_end_cuts_the_sum():
    # car's episode ends after step 0: step 0 must not count step 1's reward (a new episode)
    adv = _call(my_ppo.compute_advantages, t([[1.0], [1.0]]), t([[0.0], [0.0]]), t([[1.0], [0.0]]), t([0.0]), 1.0, 1.0)
    assert torch.allclose(torch.as_tensor(adv, dtype=torch.float32), t([[1.0], [1.0]]))


def test_advantages_discount_and_bootstrap():
    # gamma 0.5, lam 1, last_values 4: step 1 = 1 + 0.5*4 = 3; step 0 = 1 + 0.5*3 = 2.5
    adv = _call(my_ppo.compute_advantages, t([[1.0], [1.0]]), t([[0.0], [0.0]]), t([[0.0], [0.0]]), t([4.0]), 0.5, 1.0)
    assert torch.allclose(torch.as_tensor(adv, dtype=torch.float32), t([[2.5], [3.0]]))


def test_advantages_one_step_td():
    # lam 0: advantage = r + gamma*V(next) - V(now) -> [1 + .5*2 - 1, 1 + .5*3 - 2] = [1, 0.5]
    adv = _call(my_ppo.compute_advantages, t([[1.0], [1.0]]), t([[1.0], [2.0]]), t([[0.0], [0.0]]), t([3.0]), 0.5, 0.0)
    assert torch.allclose(torch.as_tensor(adv, dtype=torch.float32), t([[1.0], [0.5]]))


def test_policy_loss_clipping():
    ln = torch.log
    old = t([0.0, 0.0, 0.0])
    new = torch.stack([ln(t(1.5)), ln(t(0.5)), ln(t(0.5))])  # probability ratios 1.5, 0.5, 0.5
    adv = t([1.0, -1.0, 1.0])
    # owner's rule: ratio held in [0.8, 1.2] in every case -> objective mean(1.2, -0.8, 0.8)
    loss = _call(my_ppo.policy_loss, new, old, adv, 0.2)
    assert torch.isclose(loss, -t((1.2 - 0.8 + 0.8) / 3), atol=1e-5)
    # textbook PPO: min(ratio*adv, clipped*adv) -> the third (a mistake) gets 0.5, unclipped
    my_ppo.POLICY_LOSS_VARIANT = "textbook"
    try:
        assert torch.isclose(my_ppo.policy_loss(new, old, adv, 0.2), -t((1.2 - 0.8 + 0.5) / 3), atol=1e-5)
    finally:
        my_ppo.POLICY_LOSS_VARIANT = "owner"


def test_value_loss_mean_squared():
    loss = _call(my_ppo.value_loss, t([1.0, 3.0]), t([2.0, 3.0]))
    assert torch.isclose(loss, t(0.5)) or torch.isclose(loss, t(0.25))  # MSE, or ½·MSE
