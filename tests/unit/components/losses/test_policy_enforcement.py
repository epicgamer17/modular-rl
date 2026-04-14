import pytest
import torch
import torch.nn.functional as F
import numpy as np
from core import Blackboard
from components.losses.policy import PolicyLoss
from components.targets.formatters import OneHotPolicyTargetComponent

pytestmark = pytest.mark.unit

def test_policy_loss_enforcement_fail_on_indices():
    """PolicyLoss should now fail if targets are indices instead of distributions."""
    torch.manual_seed(42)
    blackboard = Blackboard()
    
    # Predictions: [B, T, K] - [1, 1, 3]
    preds = torch.randn(1, 1, 3)
    blackboard.predictions["policies"] = preds
    
    # Targets: [B, T] - indices
    targets = torch.tensor([[0]])
    blackboard.targets["policies"] = targets
    
    # Standard CE loss (functional)
    loss_fn = F.cross_entropy
    policy_loss_comp = PolicyLoss(loss_fn=loss_fn)
    
    # Should raise contract violation
    with pytest.raises(AssertionError, match="PolicyLoss Contract Violation"):
        policy_loss_comp.execute(blackboard)

def test_policy_loss_enforcement_success_with_one_hot_component():
    """PolicyLoss should succeed when OneHotPolicyTargetComponent is used first."""
    torch.manual_seed(42)
    blackboard = Blackboard()
    B, T, K = 2, 4, 5
    
    # 1. Setup raw action indices in data
    actions = torch.randint(0, K, (B, T))
    blackboard.data["actions"] = actions
    
    # 2. Setup predictions
    preds = torch.randn(B, T, K)
    blackboard.predictions["policies"] = preds
    
    # 3. Use OneHotPolicyTargetComponent to generate one-hot [B, T, K] distribution
    target_comp = OneHotPolicyTargetComponent(num_actions=K)
    target_comp.execute(blackboard)
    
    # 4. Use PolicyLoss (should accept the one-hot distribution)
    loss_fn = F.cross_entropy
    policy_loss_comp = PolicyLoss(loss_fn=loss_fn)
    
    # Should NOT raise AssertionError
    policy_loss_comp.execute(blackboard)
    
    assert "policy_loss" in blackboard.losses
    assert blackboard.losses["policy_loss"].item() > 0

def test_policy_loss_kl_divergence_check():
    """Ensure KL divergence is still logged correctly with the new component."""
    torch.manual_seed(42)
    blackboard = Blackboard()
    B, T, K = 1, 2, 3
    
    # Uniform policy for index 0
    actions = torch.tensor([[0, 1]])
    blackboard.data["actions"] = actions
    
    # Predictions (all zeros -> uniform after softmax)
    preds = torch.zeros(B, T, K)
    blackboard.predictions["policies"] = preds
    
    target_comp = OneHotPolicyTargetComponent(num_actions=K)
    target_comp.execute(blackboard)
    
    loss_fn = F.cross_entropy
    policy_loss_comp = PolicyLoss(loss_fn=loss_fn, log_kl=True)
    
    policy_loss_comp.execute(blackboard)
    
    # KL between one-hot [1, 0, 0] and uniform [1/3, 1/3, 1/3]
    # KL = sum(p * (log p - log q))
    # log q = log(1/3) = -1.0986
    # log p = log(1) = 0
    # KL = 1 * (0 - (-1.0986)) = 1.0986
    assert "approx_kl" in blackboard.meta
    expected_kl = -np.log(1/3)
    torch.testing.assert_close(
        torch.tensor(blackboard.meta["approx_kl"], dtype=torch.float32), 
        torch.tensor(expected_kl, dtype=torch.float32), 
        atol=1e-4, rtol=1e-4
    )
