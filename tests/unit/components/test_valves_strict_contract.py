import pytest
import torch
from components.valves import GradientScaleValve
from core import Blackboard

pytestmark = pytest.mark.unit

def test_gradient_scale_valve_strict_missing_key():
    """
    Verifies that GradientScaleValve raises KeyError if the target key 
    is missing from the blackboard predictions.
    """
    blackboard = Blackboard()
    # predictions is empty
    valve = GradientScaleValve(key="missing_tensor", scale=0.5)
    
    with pytest.raises(KeyError) as excinfo:
        valve.execute(blackboard)
    
    assert "missing_tensor" in str(excinfo.value)
    assert "[GradientScaleValve]" in str(excinfo.value)

def test_gradient_scale_valve_strict_no_grad():
    """
    Verifies that GradientScaleValve raises RuntimeError if the target tensor 
    exists but does not require gradients.
    """
    blackboard = Blackboard()
    # Tensor with requires_grad=False
    tensor = torch.randn(1, 1, requires_grad=False)
    blackboard.predictions["flat_tensor"] = tensor
    
    valve = GradientScaleValve(key="flat_tensor", scale=0.5)
    
    with pytest.raises(RuntimeError) as excinfo:
        valve.execute(blackboard)
    
    assert "detached" in str(excinfo.value)
    assert "[GradientScaleValve]" in str(excinfo.value)

def test_gradient_scale_valve_success():
    """
    Verifies that GradientScaleValve correctly registers a hook and scales 
    gradients when prerequisites are met.
    """
    torch.manual_seed(42)
    blackboard = Blackboard()
    # Tensor with requires_grad=True
    tensor = torch.randn(2, 2, requires_grad=True)
    blackboard.predictions["grad_tensor"] = tensor
    
    valve = GradientScaleValve(key="grad_tensor", scale=0.5)
    
    # Should not raise
    valve.execute(blackboard)
    
    # Trigger backward to verify hook
    loss = (tensor * 2).sum()
    loss.backward()
    
    # The raw gradient of 2*tensor is 2.0. 
    # Scaled by 0.5 via valve hook, it should be 1.0.
    assert tensor.grad is not None
    expected_grad = torch.ones_like(tensor.grad)
    assert torch.allclose(tensor.grad, expected_grad), f"Expected grad 1.0, got {tensor.grad}"
