import pytest
import torch
from torch import nn
from modules.agent_nets.modular import ModularAgentNetwork
from agents.learner.losses.shape_validator import ShapeValidator

pytestmark = pytest.mark.unit

class MockBackbone(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = (16,)
        self.fc = nn.Linear(input_shape[0], 16)
    def forward(self, x):
        return self.fc(x)

class MockHead(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.fc = nn.Linear(input_shape[0], num_actions)
        # Add representation mock for Rainbow logic
        self.representation = type('MockRep', (), {'to_expected_value': lambda self, x: x})()

    def forward(self, x):
        # Return (logits, metrics, rnn_state)
        return self.fc(x), {}, None

def test_modular_agent_network_init_no_shapes():
    """Verify ModularAgentNetwork initializes without redundant shape parameters."""
    backbone = MockBackbone((8,))
    q_head = MockHead((16,), 4)
    
    components = {
        "feature_block": backbone,
        "q_head": q_head
    }
    
    # Should not raise TypeError now
    network = ModularAgentNetwork(components=components)
    
    assert "feature_block" in network.components
    assert "q_head" in network.components

def test_modular_agent_network_learner_inference_with_validator():
    """Verify learner_inference uses the passed ShapeValidator."""
    backbone = MockBackbone((8,))
    q_head = MockHead((16,), 4)
    components = {"feature_block": backbone, "q_head": q_head}
    network = ModularAgentNetwork(components=components)
    
    # Create a batch: [B, C] -> [2, 8]
    # Rainbow logic in ModularAgentNetwork expects (B, C) and unsqueezes to (B, 1, C)
    batch = {
        "observations": torch.zeros((2, 8))
    }
    
    # Create validator: B=2, T=0 (unroll_steps), num_actions=4
    # Note: ShapeValidator in current codebase treats unroll_steps=0 as T=1 for predictions
    validator = ShapeValidator(minibatch_size=2, unroll_steps=0, num_actions=4)
    
    # This should pass validation internally
    output = network.learner_inference(batch, shape_validator=validator)
    
    assert "q_logits" in output
    # ModularAgentNetwork.learner_inference for Rainbow does Q_logits.unsqueeze(1)
    assert output["q_logits"].shape == (2, 1, 4)

def test_modular_agent_network_validation_failure():
    """Verify learner_inference fails validation if shapes are wrong."""
    backbone = MockBackbone((8,))
    q_head = MockHead((16,), 4)
    components = {"feature_block": backbone, "q_head": q_head}
    network = ModularAgentNetwork(components=components)
    
    # Batch with wrong shape: [3, 8] instead of [2, 8]
    batch = {"observations": torch.zeros((3, 8))}
    validator = ShapeValidator(minibatch_size=2, unroll_steps=0, num_actions=4)
    
    with pytest.raises(AssertionError, match="batch size mismatch"):
        network.learner_inference(batch, shape_validator=validator)
