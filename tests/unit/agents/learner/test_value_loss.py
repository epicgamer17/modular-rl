import pytest
import torch
from agents.learner.losses.value import ValueLoss, ClippedValueLoss
from agents.learner.losses.representations import ScalarRepresentation

pytestmark = pytest.mark.unit

def test_value_loss_basic():
    """
    Verifies that the standard ValueLoss correctly computes MSE.
    """
    device = torch.device("cpu")
    rep = ScalarRepresentation()
    loss_module = ValueLoss(device=device, representation=rep, target_key="returns", loss_factor=1.0)
    
    # B=2, T=1
    predictions = {"state_value": torch.tensor([[[1.0]], [[2.0]]])} # [B, T, 1]
    targets = {
        "returns": torch.tensor([[1.5], [1.5]]), # [B, T]
        "value_mask": torch.tensor([[True], [True]])
    }
    
    loss, metrics = loss_module.compute_loss(predictions, targets)
    
    # MSE = (1.0-1.5)^2 = 0.25
    # With loss_factor=1.0 -> 0.25
    expected = torch.tensor([[0.25], [0.25]])
    torch.testing.assert_close(loss, expected)

def test_clipped_value_loss_integration():
    """
    Verifies that ClippedValueLoss correctly integrates the clipping logic
    and pulls 'values' (old) from the targets dictionary.
    """
    device = torch.device("cpu")
    rep = ScalarRepresentation()
    # clip_param = 0.1, loss_factor = 1.0 for easier math
    loss_module = ClippedValueLoss(
        device=device, 
        representation=rep, 
        clip_param=0.1, 
        target_key="returns", 
        loss_factor=1.0
    )
    
    # Case: v_pred=1.4, v_old=1.0, v_target=2.0 (Should result in clipped_err = 0.81 from functional tests)
    predictions = {"state_value": torch.tensor([[[1.4]]])}
    targets = {
        "values": torch.tensor([[1.0]]),
        "returns": torch.tensor([[2.0]]),
        "value_mask": torch.tensor([[True]])
    }
    
    loss, metrics = loss_module.compute_loss(predictions, targets)
    
    # From functional test case 2: loss = 0.5 * 0.81 = 0.405
    # Multiplication by loss_factor=1.0 -> 0.405
    torch.testing.assert_close(loss, torch.tensor([[0.405]]))
    
    # Verify Shape Validation Error
    bad_targets = {
        "values": torch.tensor([[1.0, 1.0]]), # Wrong T
        "returns": torch.tensor([[2.0]]),
        "value_mask": torch.tensor([[True]])
    }
    with pytest.raises(AssertionError):
        loss_module.compute_loss(predictions, bad_targets)
