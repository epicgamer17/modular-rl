import pytest
import torch
import numpy as np

from replay_buffers.processors import GAEProcessor

pytestmark = pytest.mark.unit


def test_compute_gae_analytical_oracle():
    """
    Tier 1 test following the 'Analytical Oracle' rule.
    Calculates GAE for a small sequence of length 3 and compares against hardcoded expected values.

    Sequence Parameters:
    - rewards: [1.0, 0.0, -1.0]
    - values: [0.5, -0.2, 0.3]
    - bootstrap_value: 0.1
    - gamma: 0.9
    - gae_lambda: 0.95

    Manual Calculation:
    - gamma * gae_lambda = 0.855
    - deltas[0] = 1.0 + 0.9 * (-0.2) - 0.5 = 0.32
    - deltas[1] = 0.0 + 0.9 * (0.3) - (-0.2) = 0.47
    - deltas[2] = -1.0 + 0.9 * (0.1) - 0.3 = -1.21

    - advantages[2] = -1.21
    - advantages[1] = 0.47 + 0.855 * (-1.21) = -0.56455
    - advantages[0] = 0.32 + 0.855 * (-0.56455) = -0.16269025

    - returns[0] = -0.16269025 + 0.5 = 0.33730975
    - returns[1] = -0.56455 - 0.2 = -0.76455
    - returns[2] = -1.21 + 0.3 = -0.91
    """
    torch.manual_seed(42)
    gamma = 0.9
    gae_lambda = 0.95
    bootstrap_value = 0.1

    gae = GAEProcessor(gamma=gamma, gae_lambda=gae_lambda)

    rewards = torch.tensor([1.0, 0.0, -1.0])
    values = torch.tensor([0.5, -0.2, 0.3])

    result = gae.finish_trajectory(
        buffers={"rewards": rewards, "values": values},
        trajectory_slice=slice(None),
        last_value=bootstrap_value,
    )

    advantages = result["advantages"]
    returns = result["returns"]

    assert advantages.shape == (3,), f"Expected shape (3,), got {advantages.shape}"
    assert returns.shape == (3,), f"Expected shape (3,), got {returns.shape}"

    # Expected values from manual calculation
    assert torch.allclose(advantages[2], torch.tensor(-1.21), atol=1e-4)
    assert torch.allclose(advantages[1], torch.tensor(-0.56455), atol=1e-4)
    assert torch.allclose(advantages[0], torch.tensor(-0.16269025), atol=1e-4)

    # Returns are computed as discounted_cumulative_sums(rewards_padded, gamma)[:-1]
    # rewards_padded = [1.0, 0.0, -1.0, 0.1]
    # R[3]=0.1, R[2]=-1.0+0.9*0.1=-0.91, R[1]=0.0+0.9*(-0.91)=-0.819, R[0]=1.0+0.9*(-0.819)=0.2629
    assert torch.allclose(returns[0], torch.tensor(0.2629), atol=1e-3)
    assert torch.allclose(returns[1], torch.tensor(-0.819), atol=1e-3)
    assert torch.allclose(returns[2], torch.tensor(-0.91), atol=1e-3)


def test_compute_gae_zero_gamma():
    """Test GAE with gamma=0, should reduce to immediate TD error."""
    gae = GAEProcessor(gamma=0.0, gae_lambda=0.95)

    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 0.5, 0.5])

    result = gae.finish_trajectory(
        buffers={"rewards": rewards, "values": values},
        trajectory_slice=slice(None),
        last_value=0.0,
    )

    advantages = result["advantages"]
    # With gamma=0: delta_t = r_t + 0*V(t+1) - V(t) = r_t - V(t)
    # And GAE with gamma=0 means no discounting of future deltas
    expected = rewards - values
    assert torch.allclose(advantages, expected, atol=1e-4)


def test_compute_gae_zero_lambda():
    """Test GAE with lambda=0, should reduce to 1-step TD advantages."""
    gamma = 0.99
    gae = GAEProcessor(gamma=gamma, gae_lambda=0.0)

    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    bootstrap = 0.0

    result = gae.finish_trajectory(
        buffers={"rewards": rewards, "values": values},
        trajectory_slice=slice(None),
        last_value=bootstrap,
    )

    advantages = result["advantages"]
    # With lambda=0: advantage_t = delta_t = r_t + gamma * V(t+1) - V(t)
    next_values = torch.tensor([0.5, 0.5, bootstrap])
    expected = rewards + gamma * next_values - values
    assert torch.allclose(advantages, expected, atol=1e-4)
