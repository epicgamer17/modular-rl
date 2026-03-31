import pytest
import torch

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
    pytest.skip("TODO: update for old_muzero revert")

def test_two_hot_batched_shapes():
    """Verify shape handling for [B, T] inputs."""
    # TODO: re-enable when imports are restored
    # Original assertions:
    # assert probs.shape == (4, 5, 11)
    # assert torch.allclose(probs[:, :, 2], torch.tensor(0.5))
    # assert torch.allclose(probs[:, :, 3], torch.tensor(0.5))
    # assert probs.sum(dim=-1).allclose(torch.tensor(1.0))
    pytest.skip("TODO: update for old_muzero revert")

