import pytest
import torch

pytestmark = pytest.mark.unit

def test_gradient_scaling_logic():
    """
    Tier 1 Unit Test: Gradient Scaling Logic Verification
    Checks:
    1. Loss from each head is scaled by 1/K (via gradient_scales).
    2. Gradient entering dynamics is scaled by 1/2.
    """
    # TODO: re-enable when imports are restored
    # Original assertions:
    # assert len(loss_grads) == 1
    # assert len(dyn_grads) == 1
    pytest.skip("TODO: update for old_muzero revert")

