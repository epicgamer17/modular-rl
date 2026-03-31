import pytest
import torch
import torch.nn.functional as F

from agents.learner.losses.value import ValueLoss
from agents.learner.losses.representations import ScalarRepresentation

pytestmark = pytest.mark.unit


def test_value_loss_basic():
    """Verifies that the standard ValueLoss correctly computes MSE."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    B, T = 4, 1

    val_rep = ScalarRepresentation()
    loss_mod = ValueLoss(
        device=device,
        representation=val_rep,
        target_key="returns",
        optimizer_name="default",
        loss_factor=1.0,
        name="value_loss",
    )

    predictions = {
        "values": torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
    }  # [B, T, 1]
    targets = {
        "returns": torch.tensor([[1.5], [2.5], [3.5], [4.5]]),  # [B, T]
        "value_mask": torch.ones(B, T, dtype=torch.bool),
    }

    elementwise_loss, metrics = loss_mod.compute_loss(predictions, targets)

    # MSE between pred and target: each should be (0.5)^2 = 0.25
    assert elementwise_loss.shape == (B, T)
    assert torch.allclose(
        elementwise_loss, torch.tensor([[0.25], [0.25], [0.25], [0.25]]), atol=1e-4
    )


@pytest.mark.skip(reason="ClippedValueLoss is not yet implemented")
def test_clipped_value_loss_integration():
    """
    Verifies that ClippedValueLoss correctly integrates the clipping logic
    and pulls 'values' (old) from the targets dictionary.

    NOTE: ClippedValueLoss does not exist in the current codebase.
    This test documents expected behavior for when it is implemented.
    """
    # ClippedValueLoss is not yet implemented
    with pytest.raises(ImportError):
        from agents.learner.losses.value import ClippedValueLoss
