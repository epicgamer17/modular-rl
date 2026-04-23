import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from runtime.state import OptimizerState

# Set module level marker as per RULE[testing-standards.md]
pytestmark = pytest.mark.unit

def test_optimizer_step_changes_params():
    """
    Test 4A: Parameters change after step.
    Verifies that OptimizerState.step actually performs an optimization step.
    """
    model = nn.Linear(10, 1)
    original_weight = model.weight.clone().detach()
    
    opt = optim.SGD(model.parameters(), lr=0.1)
    opt_state = OptimizerState(opt)
    
    # Non-zero loss to ensure gradients
    x = torch.randn(4, 10)
    loss = model(x).pow(2).mean()
    
    stats = opt_state.step(loss)
    
    assert not torch.equal(model.weight, original_weight), "Parameters did not change after step"
    assert stats["loss"] >= 0
    assert stats["grad_norm"] > 0
    assert stats["lr"] == 0.1

def test_optimizer_zero_loss_no_delta():
    """
    Test 4B: Zero loss causes near-zero parameter delta.
    Verifies that if loss is zero, weights remain unchanged.
    """
    model = nn.Linear(10, 1)
    original_weight = model.weight.clone().detach()
    
    opt = optim.SGD(model.parameters(), lr=1.0)
    opt_state = OptimizerState(opt)
    
    # Zero loss that is still part of the graph
    x = torch.randn(4, 10)
    loss = model(x).pow(2).mean() * 0.0
    
    stats = opt_state.step(loss)
    
    assert torch.allclose(model.weight, original_weight, atol=1e-7), "Parameters changed despite zero loss"
    assert stats["loss"] == 0.0
    assert stats["grad_norm"] == 0.0

def test_optimizer_gradient_clipping():
    """
    Test 4C: Gradient clipping caps norm.
    Verifies that grad_clip actually limits the weight update.
    """
    # Simple model where we can predict the update
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    
    # Force a massive gradient
    # loss = w*x^2 -> d_loss/dw = 2*w*x^2
    # if w=1, x=100, grad = 20000
    x = torch.tensor([[100.0]])
    
    # Case 1: Clipping enabled
    opt_clip = optim.SGD(model.parameters(), lr=0.01)
    opt_state_clip = OptimizerState(opt_clip, grad_clip=1.0)
    
    loss = model(x).pow(2).mean()
    stats = opt_state_clip.step(loss)
    
    # If clipped to 1.0, the update should be lr * 1.0 = 0.01
    weight_after_clip = model.weight.item()
    delta_clip = abs(1.0 - weight_after_clip)
    
    # Case 2: No clipping (on a fresh model)
    model_no_clip = nn.Linear(1, 1, bias=False)
    model_no_clip.weight.data.fill_(1.0)
    opt_no_clip = optim.SGD(model_no_clip.parameters(), lr=0.01)
    opt_state_no_clip = OptimizerState(opt_no_clip, grad_clip=None)
    
    loss_no_clip = model_no_clip(x).pow(2).mean()
    opt_state_no_clip.step(loss_no_clip)
    
    weight_after_no_clip = model_no_clip.weight.item()
    delta_no_clip = abs(1.0 - weight_after_no_clip)
    
    assert delta_clip < delta_no_clip, "Clipping did not reduce the weight update"
    # With grad_clip=1.0, delta should be exactly 0.01 (lr * max_norm)
    assert pytest.approx(delta_clip, abs=1e-4) == 0.01
    assert delta_no_clip > 100.0 # Without clipping it would be huge
