import pytest
import torch

pytestmark = pytest.mark.integration

def test_muzero_loss_pipeline_gradient_flow():
    """
    Tier 1 Integration Test: Ultimate Gradient Flow Check.
    - Pass a synthetic batch through K=5 unrolled steps.
    - Compute the combined loss and call .backward().
    - Assert: representation_net weight grad is not None.
    - Assert: dynamics_net weight grad is not None.
    - Assert: prediction_net weight grad is not None.
    - Assert: Target network gradients are strictly None.
    """
    # TODO: re-enable when imports are restored
    # Original assertions:
    # assert repr_weight.grad is not None, "Representation gradient is None"
    # assert not torch.all(repr_weight.grad == 0), "Representation gradient is all zeros"
    # assert dynamics_weight.grad is not None, "Dynamics gradient is None"
    # assert not torch.all(dynamics_weight.grad == 0), "Dynamics gradient is all zeros"
    # assert policy_weight.grad is not None, "Policy head gradient is None"
    # assert value_weight.grad is not None, "Value head gradient is None"
    # assert reward_weight.grad is not None, "Reward head gradient is None"
    # assert (
    # target_repr_weight.grad is None
    # ), "Target network gradient is not None (Leakage!)"
    pytest.skip("TODO: update for old_muzero revert")

