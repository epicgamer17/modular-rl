import pytest
import torch
from modules.utils import scale_gradient

pytestmark = pytest.mark.unit

def test_muzero_dynamics_gradient_scaling():
    """
    CONTRACT: Gradients passing through this hook must be scaled by the specified factor
    during the backward pass, while the forward pass remains completely unchanged.
    """
    # Setup a dummy hidden state requiring gradients
    hidden_state = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=torch.float32)
    
    # Run the scaling hook (e.g., scale by 0.5)
    # The mathematical implementation we saw: tensor * scale + tensor.detach() * (1 - scale)
    # Forward pass: 1.0 * 0.5 + 1.0 * 0.5 = 1.0. (No change)
    scaled_hidden = scale_gradient(hidden_state, scale=0.5)
    
    # 1. Forward Pass Contract: The values must NOT change
    torch.testing.assert_close(scaled_hidden, torch.tensor([1.0, 2.0, 3.0]))
    
    # Simulate a downstream loss and backward pass
    loss = (scaled_hidden * 10.0).sum()
    loss.backward()
    
    # 2. Backward Pass Contract: 
    # d(loss)/d(scaled_hidden) = 10.0
    # scaled_hidden = hidden_state * 0.5 + const
    # d(scaled_hidden)/d(hidden_state) = 0.5
    # By chain rule: d(loss)/d(hidden_state) = 10.0 * 0.5 = 5.0
    
    expected_grad = torch.tensor([5.0, 5.0, 5.0])
    
    torch.testing.assert_close(hidden_state.grad, expected_grad)

def test_gradient_scaling_zero():
    """Verify that scale=0 effectively stops gradients."""
    hidden_state = torch.tensor([1.0, 2.0], requires_grad=True)
    scaled = scale_gradient(hidden_state, 0.0)
    
    loss = scaled.sum()
    loss.backward()
    
    assert torch.all(hidden_state.grad == 0.0)

def test_gradient_scaling_identity():
    """Verify that scale=1.0 acts as identity."""
    hidden_state = torch.tensor([1.0, 2.0], requires_grad=True)
    scaled = scale_gradient(hidden_state, 1.0)
    
    loss = (scaled * 10.0).sum()
    loss.backward()
    
    assert torch.all(hidden_state.grad == 10.0)
