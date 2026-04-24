import pytest
import torch
from agents.ppo.operators import op_ppo_objective
from core.graph import Node

pytestmark = pytest.mark.unit

class MockAC:
    def __init__(self, probs, values):
        self.probs = probs
        self.values = values
    def __call__(self, x):
        return self.probs, self.values

class MockContext:
    def __init__(self, model):
        self.model = model
    def get_model(self, handle):
        return self.model

def test_ratio_equals_one_when_same_policy():
    """Verify that ratio is exactly 1.0 when old and new log probs are identical."""
    steps = 4
    # Identical log probs
    log_probs = torch.tensor([-0.5, -1.2, -0.8, -2.0])
    # Mock model that will produce these log probs
    # We'll use a hack where ac_net returns probs that result in these log_probs
    # Actually, we can just check the math in op_ppo_objective
    # but the operator calls ac_net(obs)
    
    # Let's mock ac_net to return probs such that log_prob(action) == old_log_prob
    # For categorical, log_prob is log(p[action])
    # so p[action] = exp(log_prob)
    # actions = [0, 0, 0, 0]
    # probs = [exp(-0.5), 1-exp(-0.5)] etc.
    actions = torch.zeros(steps).long()
    probs = torch.zeros((steps, 2))
    probs[:, 0] = torch.exp(log_probs)
    probs[:, 1] = 1.0 - probs[:, 0]
    
    model = MockAC(probs, torch.zeros((steps, 1)))
    context = MockContext(model)
    
    batch = {
        "obs": torch.zeros((steps, 1)),
        "action": actions,
        "log_prob": log_probs,
        "value": torch.zeros(steps)
    }
    gae = {
        "advantages": torch.ones(steps),
        "returns": torch.zeros(steps)
    }
    
    node = Node("ppo", "PPO_Objective", params={"clip_epsilon": 0.2, "normalize_advantages": False})
    
    # We want to check the 'ratio' variable inside op_ppo_objective
    # Since it's a local variable, we'd need to mock the entire function or trust the math.
    # But the user asked for this test specifically.
    # I'll implement the test by checking the resulting loss if I can isolate actor_loss.
    # Or I can just check that the total loss is consistent with ratio=1.
    
    # If ratio=1, and advantages=1, surr1=1, surr2=1, min=1. actor_loss = -1.0.
    # If critic_loss=0 and entropy_coef=0:
    results = op_ppo_objective(node, {"batch": batch, "gae": gae}, context=context)
    loss = results["loss"]
    
    # entropy for this distribution:
    dist = torch.distributions.Categorical(probs)
    entropy = dist.entropy().mean()
    
    # expected_loss = -1.0 + 0.5 * 0.0 - 0.01 * entropy
    expected_loss = -1.0 - 0.01 * entropy
    assert loss == pytest.approx(expected_loss.item())

def test_clip_bounds_ratio():
    """Verify that ratio clipping works as expected."""
    steps = 1
    clip_epsilon = 0.2
    
    # Old log prob
    old_log_prob = torch.tensor([-1.0])
    # New log prob that would result in ratio > 1 + clip_epsilon
    # ratio = exp(new - old) => new = log(ratio) + old
    # If ratio = 2.0, new = log(2.0) - 1.0
    new_log_prob = torch.log(torch.tensor([2.0])) - 1.0
    
    actions = torch.zeros(steps).long()
    probs = torch.zeros((steps, 2))
    probs[:, 0] = torch.exp(new_log_prob)
    probs[:, 1] = 1.0 - probs[:, 0]
    
    model = MockAC(probs, torch.zeros((steps, 1)))
    context = MockContext(model)
    
    batch = {
        "obs": torch.zeros((steps, 1)),
        "action": actions,
        "log_prob": old_log_prob,
        "value": torch.zeros(steps)
    }
    # Positive advantage ensures clip is active for ratio > 1
    gae = {
        "advantages": torch.tensor([1.0]),
        "returns": torch.zeros(steps)
    }
    
    node = Node("ppo", "PPO_Objective", params={"clip_epsilon": clip_epsilon, "entropy_coef": 0, "normalize_advantages": False})
    
    results = op_ppo_objective(node, {"batch": batch, "gae": gae}, context=context)
    loss = results["loss"]
    
    # surr1 = 2.0 * 1.0 = 2.0
    # surr2 = clip(2.0, 0.8, 1.2) * 1.0 = 1.2
    # actor_loss = -min(2.0, 1.2) = -1.2
    assert loss == pytest.approx(-1.2)

def test_entropy_positive():
    """Verify that entropy is positive for a non-deterministic policy."""
    steps = 1
    # Uniform distribution (max entropy)
    probs = torch.tensor([[0.5, 0.5]])
    model = MockAC(probs, torch.zeros((steps, 1)))
    context = MockContext(model)
    
    batch = {
        "obs": torch.zeros((steps, 1)),
        "action": torch.zeros(steps).long(),
        "log_prob": torch.tensor([-0.6931]), # log(0.5)
        "value": torch.zeros(steps)
    }
    gae = {
        "advantages": torch.zeros(steps),
        "returns": torch.zeros(steps)
    }
    
    node = Node("ppo", "PPO_Objective", params={"clip_epsilon": 0.2, "entropy_coef": 1.0, "critic_coef": 0, "normalize_advantages": False})
    
    results = op_ppo_objective(node, {"batch": batch, "gae": gae}, context=context)
    loss = results["loss"]
    
    # ratio=1, adv=0 => actor_loss = 0
    # critic_loss = 0 (next_values and returns are 0)
    # loss = 0 + 0 - 1.0 * entropy = -entropy
    # For [0.5, 0.5], entropy = - (0.5*log(0.5) + 0.5*log(0.5)) = -log(0.5) = log(2) ≈ 0.6931
    assert loss.item() < 0 # -entropy should be negative
    assert loss == pytest.approx(-0.6931, abs=1e-4)
