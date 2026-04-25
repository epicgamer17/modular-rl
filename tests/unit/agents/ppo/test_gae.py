import pytest
import torch
import numpy as np
from ops.rl.advantage import op_gae
from core.graph import Node

pytestmark = pytest.mark.unit

def reference_gae(rewards, values, terminateds, next_values, gamma, gae_lambda):
    """NumPy reference implementation of GAE."""
    advantages = np.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        non_terminal = 1.0 - terminateds[t]
        delta = rewards[t] + gamma * next_values[t] * non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
    return advantages

class MockModel:
    def __init__(self, values, next_values):
        self.values = values
        self.next_values = next_values
        self.params = {}

    def __call__(self, x):
        # In our test, x contains indices. 
        # obs are 0..N-1, next_obs are also 0..N-1 but we want them to return next_values.
        # We'll use a hack: if the first value is 0.0 and it's a batch, we'll check if it's next_obs.
        # Better: just use different ranges for obs and next_obs.
        indices = x.view(-1).long()
        # If any index is >= 100, it's a next_obs (we'll offset them in the test)
        is_next = (indices >= 100).any()
        if is_next:
            vals = self.next_values[indices - 100]
        else:
            vals = self.values[indices]
            
        if x.dim() == 2:
            return torch.zeros((len(indices), 2)), vals.unsqueeze(1)
        else:
            return torch.zeros((1, 2)), vals.unsqueeze(0)
    
    def named_parameters(self):
        return {}

class MockContext:
    def __init__(self, model):
        self.model = model
    def get_model(self, handle):
        return self.model

def test_gae_matches_reference():
    """Verify that op_gae matches the reference NumPy implementation."""
    steps = 10
    gamma = 0.99
    gae_lambda = 0.95
    
    rewards = torch.randn(steps)
    values = torch.randn(steps)
    terminateds = torch.zeros(steps)
    terminateds[5] = 1.0 # One terminal state in the middle
    next_values = torch.randn(steps)
    
    # Obs and next_obs will be indices for our mock model
    obs = torch.arange(steps).unsqueeze(1).float()
    next_obs = (torch.arange(steps) + 100).unsqueeze(1).float()
    
    from core.batch import TransitionBatch
    batch = TransitionBatch(
        obs=obs,
        reward=rewards,
        terminated=terminateds,
        truncated=torch.zeros_like(terminateds),
        next_obs=next_obs,
        action=torch.zeros(steps),
        value=values
    )
    
    node = Node(node_id="gae", node_type="PPO_GAE", params={"gamma": gamma, "gae_lambda": gae_lambda})
    
    # New op_gae requires next_value and next_terminated
    # In PPO, this usually comes from the value at T
    inputs = {
        "batch": batch,
        "next_value": torch.tensor([0.0]), # Placeholder since it's not strictly needed for reference check if T is small
        "next_terminated": torch.tensor([False])
    }
    
    result = op_gae(node, inputs, context=None)
    advantages = result["advantages"]
    
    # Reference needs the actual next values for each step for the recursive formula
    # but op_gae bootstraps from batch.value[t+1] or next_value.
    # Our reference_gae expects next_values array.
    ref_advantages = reference_gae(
        rewards.numpy(), 
        values.numpy(), 
        terminateds.numpy(), 
        # For the last step, it uses next_value
        np.concatenate([values[1:].numpy(), [0.0]]), 
        gamma, 
        gae_lambda
    )
    
    assert torch.allclose(advantages, torch.from_numpy(ref_advantages), atol=1e-6)

def test_gae_handles_truncation():
    """Verify that GAE correctly bootstraps on truncation (timeout masking)."""
    steps = 2
    gamma = 0.9
    gae_lambda = 1.0 # Lambda=1 reduces to standard advantage
    
    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.5, 0.5])
    terminateds = torch.tensor([0.0, 0.0]) # No termination
    truncateds = torch.tensor([0.0, 1.0]) # Truncated at the end
    next_values = torch.tensor([0.5, 10.0]) # Large next value on truncation
    
    model = MockModel(values, next_values)
    context = MockContext(model)
    
    batch = {
        "obs": torch.arange(steps).unsqueeze(1).float(),
        "reward": rewards,
        "terminated": terminateds,
        "truncated": truncateds,
        "next_obs": (torch.arange(steps) + 100).unsqueeze(1).float()
    }
    
    node = Node("gae", "PPO_GAE", params={"gamma": gamma, "gae_lambda": gae_lambda})
    
    result = op_gae(node, {"batch": batch}, context=context)
    advantages = result["advantages"]
    
    # For t=1 (last step):
    # delta_1 = r_1 + gamma * V(s_2) * (1 - term_1) - V(s_1)
    # term_1 is 0 (it was truncated, not terminated)
    # delta_1 = 1.0 + 0.9 * 10.0 * 1.0 - 0.5 = 9.5
    assert advantages[1] == pytest.approx(9.5)
    
    # For t=0:
    # delta_0 = r_0 + gamma * V(s_1) * (1 - term_0) - V(s_0) = 1.0 + 0.9 * 0.5 * 1.0 - 0.5 = 0.95
    # A_0 = delta_0 + gamma * lam * (1 - term_0) * A_1 = 0.95 + 0.9 * 1.0 * 1.0 * 9.5 = 0.95 + 8.55 = 9.5
    assert advantages[0] == pytest.approx(9.5)
