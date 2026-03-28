import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from search.search_py.prior_injectors import DirichletInjector

pytestmark = pytest.mark.unit

def test_dirichlet_noise_injection_math():
    """
    MATH: Noise injection must correctly interpolate between the prior policy
    and the sampled Dirichlet noise using the exploration_fraction.
    
    Formula: p_noisy = (1 - fraction) * p_prior + fraction * noise
    """
    # Setup inputs: 2 legal moves [0, 1] among 3 possible actions
    policy = torch.tensor([0.8, 0.2, 0.0], dtype=torch.float32)
    legal_moves = [0, 1]
    
    # Mock config
    config = MagicMock()
    config.use_dirichlet = True
    config.dirichlet_alpha = 0.3
    config.dirichlet_fraction = 0.25
    
    # Mock Dirichlet sampling to return deterministic values: [0.1, 0.9] for legal moves
    mock_noise = np.array([0.1, 0.9])
    
    with patch('numpy.random.dirichlet', return_value=mock_noise):
        injector = DirichletInjector()
        noisy_policy = injector.inject(
            policy,
            legal_moves=legal_moves,
            config=config,
            exploration=True
        )
        
        # Hand calculation:
        # Action 0: (1 - 0.25) * 0.8 + (0.25 * 0.1) = 0.600 + 0.025 = 0.625
        # Action 1: (1 - 0.25) * 0.2 + (0.25 * 0.9) = 0.150 + 0.225 = 0.375
        # Action 2: (1 - 0.25) * 0.0 + (0.25 * 0) = 0.0 (untouched but effectively 0)
        
        expected_probs = torch.tensor([0.625, 0.375, 0.0], dtype=torch.float32)
        
        torch.testing.assert_close(noisy_policy, expected_probs)
        
        # Contract: The new priors must still sum to exactly 1.0
        assert torch.allclose(noisy_policy.sum(), torch.tensor(1.0))

def test_dirichlet_noise_injection_skips_when_disabled():
    """Verify that injection is skipped if DISABLED in config or exploration is False."""
    policy = torch.tensor([0.8, 0.2], dtype=torch.float32)
    legal_moves = [0, 1]
    
    config = MagicMock()
    config.use_dirichlet = False
    
    injector = DirichletInjector()
    
    # If disabled in config
    noisy_policy = injector.inject(policy, legal_moves, config, exploration=True)
    assert torch.equal(noisy_policy, policy)
    
    # If exploration is False
    config.use_dirichlet = True
    noisy_policy = injector.inject(policy, legal_moves, config, exploration=False)
    assert torch.equal(noisy_policy, policy)
