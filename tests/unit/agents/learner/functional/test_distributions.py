import pytest
import torch
from agents.learner.functional.distributions import project_scalars_to_discrete_support

pytestmark = pytest.mark.unit

def test_two_hot_categorical_encoding():
    """
    MATH: A scalar value must distribute its probability mass exactly 
    between the two nearest bins in the support vector.
    
    Setup:
    - Support: [-2.0, -1.0, 0.0, 1.0, 2.0]
    - vmin = -2.0, vmax = 2.0, bins = 5
    - delta_z = 1.0
    """
    vmin = -2.0
    vmax = 2.0
    bins = 5
    
    # Test values: exactly on a bin, between bins, and out of bounds
    scalars = torch.tensor([
        0.0,    # Exactly on index 2 (value 0.0)
        0.4,    # Between 0 (idx 2) and 1 (idx 3). 60% on 0, 40% on 1.
        -1.75,  # Between -2 (idx 0) and -1 (idx 1). 
                # p = (-1.75 - (-2)) / 1 = 0.25. 
                # 75% on -2 (idx 0), 25% on -1 (idx 1).
        5.0     # Out of bounds (clamps to index 4)
    ])

    # expected_probs shape: [batch_size, support_size]
    expected_probs = torch.tensor([
        [0.0,  0.0,  1.0,  0.0,  0.0],
        [0.0,  0.0,  0.6,  0.4,  0.0],
        [0.75, 0.25, 0.0,  0.0,  0.0], # Corrected from user prompt which was swapped
        [0.0,  0.0,  0.0,  0.0,  1.0]
    ])

    # Run system under test
    probabilities = project_scalars_to_discrete_support(scalars, vmin, vmax, bins)
    
    torch.testing.assert_close(probabilities, expected_probs)

def test_two_hot_batched_shapes():
    """Verify shape handling for [B, T] inputs."""
    vmin, vmax, bins = 0.0, 10.0, 11
    scalars = torch.ones((4, 5)) * 2.5 # [B, T]
    
    probs = project_scalars_to_discrete_support(scalars, vmin, vmax, bins)
    assert probs.shape == (4, 5, 11)
    
    # 2.5 should be split between idx 2 and 3
    assert torch.allclose(probs[:, :, 2], torch.tensor(0.5))
    assert torch.allclose(probs[:, :, 3], torch.tensor(0.5))
    assert probs.sum(dim=-1).allclose(torch.tensor(1.0))
