import pytest
import torch
import torch.nn as nn
from runtime.state import OptimizerState

pytestmark = pytest.mark.unit

def test_grad_clip_caps_norm():
    """
    Test that OptimizerState correctly caps the gradient norm.
    """
    torch.manual_seed(42)
    
    # Simple model
    model = nn.Linear(10, 1)
    # Huge gradient
    loss_fn = lambda x: x.sum() * 1000.0
    
    # Optimizer with clipping
    max_norm = 0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt_state = OptimizerState(optimizer, grad_clip=max_norm)
    
    # Input
    x = torch.randn(1, 10)
    output = model(x)
    loss = loss_fn(output)
    
    # Step returns metrics
    metrics = opt_state.step(loss)
    
    # Check that grad_norm in metrics is the norm BEFORE clipping (as per torch.nn.utils.clip_grad_norm_)
    # but the actual gradients in the model should be clipped.
    
    # Actually, clip_grad_norm_ returns the TOTAL norm before clipping.
    assert metrics["grad_norm"] > max_norm, "Initial grad norm should be large"
    
    # Verify that gradients are clipped
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # The total norm should now be <= max_norm (with some floating point tolerance)
    assert total_norm <= max_norm + 1e-6, f"Gradients were not clipped correctly: {total_norm} > {max_norm}"
