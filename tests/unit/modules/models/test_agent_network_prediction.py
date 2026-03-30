import pytest
import torch
import torch.nn as nn

from modules.models.agent_network import AgentNetwork
from agents.factories.model import build_agent_network
from configs.agents.muzero import MuZeroConfig

pytestmark = pytest.mark.unit

def test_prediction_backbone_integration(make_muzero_config_dict, cartpole_game_config):
    """
    Ensures that the prediction_backbone is correctly parsed by `build_agent_network`
    and invoked inside `AgentNetwork` during inference.
    Old MuZero relied on a prediction tower after the world model latents.
    """
    torch.manual_seed(42)
    
    # 1. Create a config with a specific prediction backbone size
    cfg = make_muzero_config_dict(
        representation_backbone={"type": "mlp", "widths": [16]},
        prediction_backbone={"type": "mlp", "widths": [32]},
        value_head={"output_strategy": {"type": "scalar"}, "neck": {"type": "identity"}},
        policy_head={"output_strategy": {"type": "categorical"}, "neck": {"type": "identity"}},
    )
    muzero_config = MuZeroConfig(cfg, cartpole_game_config)

    # 2. Build the network
    network = build_agent_network(
        config=muzero_config,
        obs_dim=(4,),
        num_actions=cartpole_game_config.num_actions,
    )

    # 3. Verify it exists
    assert "prediction" in network.components, ("prediction_backbone was not added to AgentNetwork components. "
                                                "It is required to maintain architectural parity with older MuZero variants.")

    # 4. Verify shapes and parameters
    pred_layer = network.components["prediction"]
    assert hasattr(pred_layer, "output_shape"), "Prediction backbone must expose output_shape."
    assert pred_layer.output_shape == (32,), f"Expected output_shape (32,), got {pred_layer.output_shape}."

    # 5. Verify it runs in obs_inference without crashing
    obs = torch.rand(2, 4)
    out = network.obs_inference(obs)

    assert out.value is not None
    assert out.policy is not None
    assert out.policy.logits.shape == (2, cartpole_game_config.num_actions)

def test_agent_network_prediction_routing():
    """
    Test routing of purely mocked prediction tower inside AgentNetwork's `_apply_spatial_temporal`.
    """
    class MockPrediction(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_shape = (128,)
            self.called = False

        def forward(self, x):
            self.called = True
            # Expand feature dimension from 64 to 128 to prove it was applied
            return torch.cat([x, x], dim=-1)

    network = AgentNetwork(
        input_shape=(64,),
        num_actions=2,
        prediction_fn=lambda input_shape: MockPrediction()
    )

    tensor = torch.ones(2, 64)
    # The _apply_spatial_temporal method takes [B*T, D] originally or flat_x,
    # wait: it takes `tensor` as `latent.unsqueeze(1)` normally, so shape is [B, T, D]
    tensor_seq = tensor.unsqueeze(1)
    
    out, next_state = network._apply_spatial_temporal(tensor_seq, 2, 1, state={})
    
    assert "prediction" in network.components
    assert network.components["prediction"].called, "Prediction backbone was not called during spatial-temporal application."
    assert out.shape == (2, 128), "Prediction backbone did not modify output shape correctly."
