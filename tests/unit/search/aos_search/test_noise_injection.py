import pytest
import torch
from unittest.mock import patch
from search.aos_search.functional_modifiers import apply_dirichlet_noise

pytestmark = pytest.mark.unit

def test_root_node_dirichlet_noise_injection():
    """
    MATH: Noise injection must correctly interpolate between the prior policy
    and the sampled Dirichlet noise using the exploration_fraction.
    
    Formula: p_noisy = (1 - fraction) * p_legal + fraction * noise
    """
    # Setup inputs
    # Child 0 prior = 0.8. Child 1 prior = 0.2
    # Logits: log(0.8) approx -0.2231, log(0.2) approx -1.6094
    logits = torch.tensor([[torch.log(torch.tensor(0.8)), torch.log(torch.tensor(0.2))]], dtype=torch.float32)
    
    alpha = 0.3
    exploration_fraction = 0.25
    
    # Mock Dirichlet sampling to return deterministic values: [0.1, 0.9]
    mock_noise = torch.tensor([[0.1, 0.9]], dtype=torch.float32)
    
    with patch('torch.distributions.Dirichlet.sample', return_value=mock_noise):
        # Run system under test
        noisy_logits = apply_dirichlet_noise(
            logits, 
            alpha=alpha, 
            fraction=exploration_fraction
        )
        
        # Convert back to probabilities for easy verification
        noisy_probs = torch.exp(noisy_logits)
        
        # Hand calculation:
        # Child 0: (1 - 0.25) * 0.8 + (0.25 * 0.1) = 0.600 + 0.025 = 0.625
        # Child 1: (1 - 0.25) * 0.2 + (0.25 * 0.9) = 0.150 + 0.225 = 0.375
        
        expected_probs = torch.tensor([[0.625, 0.375]], dtype=torch.float32)
        
        torch.testing.assert_close(noisy_probs, expected_probs)
        
        # Contract: The new priors must still sum to exactly 1.0
        assert torch.allclose(noisy_probs.sum(dim=-1), torch.tensor([1.0]))

def test_dirichlet_noise_with_masking():
    """Verify that noise does not leak into illegal actions."""
    # [B=1, A=3]
    logits = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    # Mask: only actions 0 and 2 are legal
    valid_mask = torch.tensor([[True, False, True]], dtype=torch.bool)
    
    # Simple uniform noise for 3 actions: [0.2, 0.5, 0.3]
    # After masking and renormalizing legal ones:
    # Action 0: 0.2. Action 2: 0.3. Sum: 0.5.
    # Renormalized: Action 0: 0.4. Action 2: 0.6.
    mock_noise = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float32)
    
    exploration_fraction = 0.5
    
    with patch('torch.distributions.Dirichlet.sample', return_value=mock_noise):
        noisy_logits = apply_dirichlet_noise(
            logits, 
            alpha=0.3, 
            fraction=exploration_fraction,
            valid_mask=valid_mask
        )
        
        noisy_probs = torch.exp(noisy_logits)
        
        # Action 1 must be exactly 0
        assert noisy_probs[0, 1] == 0.0
        assert noisy_logits[0, 1] == -float('inf')
        
        # Legal probs: p_legal = [0.5, 0, 0.5] (softmax of [0, -inf, 0])
        # Noise (masked & renorm): [0.4, 0, 0.6]
        # Blended: 0.5 * [0.5, 0, 0.5] + 0.5 * [0.4, 0, 0.6] 
        # = [0.25 + 0.20, 0, 0.25 + 0.30] = [0.45, 0, 0.55]
        
        expected_probs = torch.tensor([[0.45, 0.0, 0.55]], dtype=torch.float32)
        torch.testing.assert_close(noisy_probs, expected_probs)
