import pytest
import torch
import torch.nn as nn
from agents.learner.base import UniversalLearner
from agents.learner.losses import LossPipeline

pytestmark = pytest.mark.unit

class MockNetwork(nn.Module):
    def __init__(self, input_shape=(4,), num_actions=2):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.fc = nn.Linear(input_shape[0], num_actions)
        
    def learner_inference(self, batch):
        obs = batch["observations"]
        logits = self.fc(obs)
        return {"logits": logits}

class MockLossPipeline:
    def run(self, predictions, targets, weights, gradient_scales=None):
        loss_tensor = predictions["logits"].sum()
        return {"default": loss_tensor}, {"total_loss": loss_tensor.item()}, None

def test_universal_learner_init_no_config():
    """Verify UniversalLearner initializes correctly without a config object."""
    net = MockNetwork()
    device = torch.device("cpu")
    
    learner = UniversalLearner(
        agent_network=net,
        device=device,
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        loss_pipeline=MockLossPipeline(),
        optimizer=torch.optim.Adam(net.parameters(), lr=1e-3),
        gradient_accumulation_steps=2,
        max_grad_norm=0.5
    )
    
    assert learner.gradient_accumulation_steps == 2
    assert learner.max_grad_norm == 0.5
    assert not hasattr(learner, "config")

def test_universal_learner_step_no_config():
    """Verify UniversalLearner.step runs without a config object."""
    net = MockNetwork()
    device = torch.device("cpu")
    loss_pipeline = MockLossPipeline()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    learner = UniversalLearner(
        agent_network=net,
        device=device,
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        loss_pipeline=loss_pipeline,
        optimizer=optimizer,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0
    )
    
    batch = {
        "observations": torch.randn(2, 4), # B=2, T=1 (implied)
    }
    # Mocking B, T check in learner._build_step_metrics
    # Original code: B, T = any_pred.shape[:2]
    # We need to make sure predictions have at least 2 dimensions
    
    batch_iterator = [batch]
    
    metrics_list = list(learner.step(batch_iterator))
    assert len(metrics_list) > 0
    assert "total_loss" in metrics_list[0]

if __name__ == "__main__":
    test_universal_learner_init_no_config()
    test_universal_learner_step_no_config()
    print("All tests passed!")
