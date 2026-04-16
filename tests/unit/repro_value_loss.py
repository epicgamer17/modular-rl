import torch
import pytest
import sys
import os

# Add project root to path for imports
sys.path.append(os.getcwd())

from components.losses.value import ClippedValueLoss, ScalarValueLoss
from core import Blackboard

def test_clipped_value_loss_shape_alignment_repro():
    """
    Reproduce the shape mismatch bug in ClippedValueLoss.
    Bug: values is [B, 1, 1], returns is [B, 1].
    Subtraction [B, 1, 1] - [B, 1] broadcasts to [B, B, 1] if not aligned.
    """
    B, T = 4, 1
    clip_param = 0.2
    
    # Buggy scenario:
    # predictions.values: [B, T, 1] -> [4, 1, 1]
    # targets.returns: [B, T] -> [4, 1]
    values = torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]]]) # [4, 1, 1]
    returns = torch.tensor([[1.0], [2.0], [3.0], [4.0]])        # [4, 1]
    old_values = torch.tensor([[1.0], [2.0], [3.0], [4.0]])    # [4, 1]
    
    blackboard = Blackboard()
    blackboard.predictions["values"] = values
    blackboard.targets["returns"] = returns
    blackboard.targets["values"] = old_values # usually named 'values' in PPO data

    loss_comp = ClippedValueLoss(
        clip_param=clip_param,
        target_key="targets.returns",
        old_values_key="targets.values"
    )

    try:
        output = loss_comp.execute(blackboard)
        elementwise_loss = output["meta.elementwise_losses.value_loss"]
        
        # Check the shape.
        assert elementwise_loss.shape == (B, T, 1), f"Expected shape (4, 1, 1), got {elementwise_loss.shape}"
        
        # Check values: if it was pairwise [4, 4, 1], the mean would be different.
        assert torch.allclose(elementwise_loss, torch.zeros_like(elementwise_loss)), "Loss should be zero for matching targets"

    except AssertionError as e:
        if "shape mismatch" in str(e) or "internal shape error" in str(e):
             print(f"Bug Reproduced: ClippedValueLoss failed with shape mismatch: {e}")
             return
        raise e
    except Exception as e:
        print(f"ClippedValueLoss failed with unexpected error: {e}")
        raise e

def test_scalar_value_loss_name_error_repro():
    """Reproduce NameError in ScalarValueLoss where B, T are used before definition."""
    preds = torch.randn(4, 1, 1)
    targets = torch.randn(4, 1) # Mismatch triggers the reshape logic
    
    blackboard = Blackboard()
    blackboard.predictions["values"] = preds
    blackboard.targets["values"] = targets
    
    loss_comp = ScalarValueLoss(target_key="targets.values")
    
    try:
        loss_comp.execute(blackboard)
    except NameError as e:
        if "name 'B' is not defined" in str(e):
             print(f"ScalarValueLoss NameError reproduced: {e}")
             return
        raise e
    except Exception as e:
        print(f"ScalarValueLoss failed with unexpected error types: {type(e).__name__}: {e}")
        raise e

if __name__ == "__main__":
    print("Running ClippedValueLoss repro...")
    test_clipped_value_loss_shape_alignment_repro()
    
    print("\nRunning ScalarValueLoss repro...")
    try:
        test_scalar_value_loss_name_error_repro()
    except Exception as e:
        print(f"Test failed: {e}")
