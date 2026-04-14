import pytest
import torch
import numpy as np
from core import Blackboard
from components.targets.formatters import OneHotPolicyTargetComponent

pytestmark = pytest.mark.unit

def test_one_hot_policy_target_component_standard():
    """Test standard conversion from [B, T] indices to [B, T, K] one-hot."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    num_actions = 5
    B, T = 8, 10
    
    component = OneHotPolicyTargetComponent(num_actions=num_actions)
    blackboard = Blackboard()
    
    # Mock data: actions as indices [B, T]
    actions = torch.randint(0, num_actions, (B, T))
    blackboard.data["actions"] = actions
    
    component.execute(blackboard)
    
    # Check output
    assert "policies" in blackboard.targets
    policies = blackboard.targets["policies"]
    
    assert policies.shape == (B, T, num_actions)
    assert policies.dtype == torch.float32
    
    # Verify one-hotness
    # For each (b, t), only one index should be 1.0, rest 0.0
    for b in range(B):
        for t in range(T):
            idx = actions[b, t].item()
            expected = torch.zeros(num_actions)
            expected[idx] = 1.0
            torch.testing.assert_close(policies[b, t], expected)

def test_one_hot_policy_target_component_unsqeezed():
    """Test conversion when indices are [B, T, 1]."""
    torch.manual_seed(42)
    
    num_actions = 3
    B, T = 4, 5
    
    component = OneHotPolicyTargetComponent(num_actions=num_actions)
    blackboard = Blackboard()
    
    # [B, T, 1] shape
    actions = torch.randint(0, num_actions, (B, T, 1))
    blackboard.data["actions"] = actions
    
    component.execute(blackboard)
    
    policies = blackboard.targets["policies"]
    assert policies.shape == (B, T, num_actions)

def test_one_hot_policy_target_custom_keys():
    """Test with custom source and destination keys."""
    num_actions = 10
    component = OneHotPolicyTargetComponent(
        num_actions=num_actions, 
        source_key="data.my_actions", 
        dest_key="my_policy"
    )
    blackboard = Blackboard()
    
    actions = torch.tensor([[0, 1], [9, 5]]) # [2, 2]
    blackboard.data["my_actions"] = actions
    
    component.execute(blackboard)
    
    assert "my_policy" in blackboard.targets
    assert blackboard.targets["my_policy"].shape == (2, 2, 10)
