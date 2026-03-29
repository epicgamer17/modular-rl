import pytest
import numpy as np
import torch
from agents.learner.functional.advantages import compute_gae

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
    # 1. Input Setup
    rewards = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    values = np.array([0.5, -0.2, 0.3], dtype=np.float32)
    bootstrap_value = 0.1
    gamma = 0.9
    gae_lambda = 0.95
    
    # 2. Expected Output (Analytical Oracle)
    expected_adv = np.array([-0.16269025, -0.56455, -1.21], dtype=np.float32)
    expected_ret = np.array([0.33730975, -0.76455, -0.91], dtype=np.float32)
    
    # 3. Execution
    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        gamma=gamma,
        gae_lambda=gae_lambda
    )
    
    # 4. Assertions
    # Shape Contract
    assert advantages.shape == (3,), f"Expected shape (3,), got {advantages.shape}"
    assert returns.shape == (3,), f"Expected shape (3,), got {returns.shape}"
    
    # Analytical Oracle Contract
    np.testing.assert_allclose(advantages, expected_adv, rtol=1e-5, err_msg="Advantages do not match analytical oracle")
    np.testing.assert_allclose(returns, expected_ret, rtol=1e-5, err_msg="Returns do not match analytical oracle")

def test_compute_gae_zero_gamma():
    """Test GAE with gamma=0, should reduce to immediate TD error."""
    rewards = np.array([1.0, 2.0], dtype=np.float32)
    values = np.array([0.5, 0.5], dtype=np.float32)
    bootstrap_value = 0.1
    gamma = 0.0
    gae_lambda = 0.95
    
    # delta_0 = 1.0 + 0.0 * 0.5 - 0.5 = 0.5
    # delta_1 = 2.0 + 0.0 * 0.1 - 0.5 = 1.5
    # advantages[1] = 1.5
    # advantages[0] = 0.5
    expected_adv = np.array([0.5, 1.5], dtype=np.float32)
    
    advantages, _ = compute_gae(rewards, values, bootstrap_value, gamma, gae_lambda)
    np.testing.assert_allclose(advantages, expected_adv)

def test_compute_gae_zero_lambda():
    """Test GAE with lambda=0, should reduce to 1-step TD advantages."""
    rewards = np.array([1.0, 0.0], dtype=np.float32)
    values = np.array([0.1, 0.2], dtype=np.float32)
    bootstrap_value = 0.3
    gamma = 1.0
    gae_lambda = 0.0
    
    # deltas:
    # d0 = 1.0 + 1.0*0.2 - 0.1 = 1.1
    # d1 = 0.0 + 1.0*0.3 - 0.2 = 0.1
    # discount = gamma * lambda = 0
    # advs = [1.1, 0.1]
    expected_adv = np.array([1.1, 0.1], dtype=np.float32)
    
    advantages, _ = compute_gae(rewards, values, bootstrap_value, gamma, gae_lambda)
    np.testing.assert_allclose(advantages, expected_adv)
