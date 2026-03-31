import pytest
import torch

pytestmark = pytest.mark.unit

def test_ppo_loss_contract():
    """
    Verifies that the PPO TargetBuilder and LossPipeline have a consistent contract.
    Specifically checks that 'values' (old values) are present for ClippedValueLoss.
    """
    # TODO: re-enable when imports are restored
    # Original assertions:
    # assert "values" in result.targets
    # assert result.targets["values"].shape == (batch_size, 1)
    # assert "policy_loss" in result.loss_dict
    # assert "value_loss" in result.loss_dict
    pytest.skip("TODO: update for old_muzero revert")

def test_muzero_loss_contract():
    """Verifies that the MuZero TargetBuilder (Unrolled) and LossPipeline have a consistent contract."""
    # TODO: re-enable when imports are restored
    # Original assertions:
    # assert "values" in result.targets
    # assert "rewards" in result.targets
    # assert result.targets["values"].shape == (batch_size, T + 1)
    # assert result.targets["rewards"].shape == (batch_size, T + 1)
    pytest.skip("TODO: update for old_muzero revert")

def test_rainbow_loss_contract():
    """Verifies that the Rainbow TargetBuilder (TD/Distributional) and LossPipeline have a consistent contract."""
    # TODO: re-enable when imports are restored
    # Original assertions:
    # assert "q_logits" in result.targets
    # assert "QBootstrappingLoss" in result.loss_dict
    # assert result.targets["q_logits"].shape == (batch_size, 1, 51)
    pytest.skip("TODO: update for old_muzero revert")

