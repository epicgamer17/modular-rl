import torch
import torch.nn as nn
import pytest
from typing import Tuple, Dict, Any, Callable

# Project Imports
from modules.models.agent_network import AgentNetwork
from modules.models.world_model import WorldModel
from modules.heads.policy import PolicyHead
from modules.heads.value import ValueHead
from agents.learner.losses.representations import ClassificationRepresentation, DiscreteSupportRepresentation

pytestmark = pytest.mark.integration

class SimpleBackbone(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], output_channels: int = 8):
        super().__init__()
        self.input_shape = input_shape
        # We named the layer 'conv' as used in our previous integration test for consistency
        self.conv = nn.Conv2d(input_shape[0], output_channels, 3, padding=1)
        self.output_shape = (output_channels, *input_shape[1:])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

def build_test_muzero_network(C, H, W, num_actions, latent_channels=8):
    """Minimal factory to build a MuZero-compatible AgentNetwork for testing."""
    def repr_fn(input_shape: Tuple[int, ...]) -> nn.Module:
        return SimpleBackbone(input_shape, output_channels=latent_channels)

    def world_model_fn(latent_dimensions, num_actions, num_players):
        def dynamics_fn(input_shape):
            m = nn.Conv2d(input_shape[0], latent_dimensions[0], 3, padding=1)
            m.output_shape = latent_dimensions
            return m
        return WorldModel(
            latent_dimensions=latent_dimensions,
            num_actions=num_actions,
            num_players=num_players,
            dynamics_fn=dynamics_fn,
            env_head_fns={
                "reward_logits": lambda input_shape, **kwargs: ValueHead(
                    input_shape=input_shape,
                    representation=DiscreteSupportRepresentation(vmin=-1, vmax=1, bins=3),
                    name="reward_logits"
                )
            }
        )

    head_fns = {
        "policy_logits": lambda input_shape, num_actions, **kwargs: PolicyHead(
            input_shape=input_shape,
            representation=ClassificationRepresentation(num_actions),
            name="policy_logits"
        ),
        "state_value": lambda input_shape, **kwargs: ValueHead(
            input_shape=input_shape,
            representation=DiscreteSupportRepresentation(vmin=-1, vmax=1, bins=3),
            name="state_value"
        )
    }

    return AgentNetwork(
        input_shape=(C, H, W),
        num_actions=num_actions,
        representation_fn=repr_fn,
        world_model_fn=world_model_fn,
        head_fns=head_fns
    )

def test_muzero_gradient_flow_and_target_stop():
    """
    Ensure the Dynamics and Prediction networks receive gradients, 
    while the Target/Momentum network remains isolated.
    """
    B, K = 2, 2
    C, H, W = 3, 8, 8
    num_actions = 4
    
    # 1. Instantiate Main (Online) Agent and Target (Momentum) Agent
    agent = build_test_muzero_network(C, H, W, num_actions)
    target_agent = build_test_muzero_network(C, H, W, num_actions)
    
    # Explicitly clear any residual gradients
    agent.zero_grad()
    target_agent.zero_grad()
    
    # 2. Fake Forward Pass (Online)
    obs = torch.randn(B, C, H, W)
    actions = torch.randint(0, num_actions, (B, K))
    
    # Initial representation check
    root_out = agent.obs_inference(obs)
    # Dynamics unroll check (K steps)
    out1 = agent.hidden_state_inference(root_out.recurrent_state, actions[:, 0])
    
    # 3. Dummy Loss Calculation
    # We take the value prediction from step 1 (requires Representation -> Dynamics -> ValueHead)
    pred_v1 = out1.value
    
    # 4. Target Generation (Isolation Check)
    # Simulate an EfficientZero-style consistency target or MCTS value lookup
    # using the separate target_agent network.
    with torch.no_grad():
        target_out = target_agent.obs_inference(obs)
        target_v0 = target_out.value.detach()
    
    # Calculate MSE Loss between pred_v1 (flow) and target_v0 (stop)
    loss = torch.nn.functional.mse_loss(pred_v1, target_v0)
    loss.backward()
    
    # --- ASSERTIONS ---
    
    # A. Verify Gradient Flow through Online Network
    # 1. Did it reach the Representation Backbone?
    repr_backbone = agent.components["representation"].conv
    assert repr_backbone.weight.grad is not None, "Gradient failed to reach the Representation backbone."
    assert not torch.all(repr_backbone.weight.grad == 0), "Gradient in Representation backbone is exactly zero."
    
    # 2. Did it flow through the Dynamics Pipeline?
    # This proves the loss on v1 successfully traveled through the dynamics unroll step
    dynamics_conv = agent.components["world_model"].dynamics_pipeline.dynamics
    assert dynamics_conv.weight.grad is not None, "Gradient failed to flow back through the Dynamics network."
    assert not torch.all(dynamics_conv.weight.grad == 0), "Gradient in Dynamics network is exactly zero."
    
    # 3. Did it reach the Value Head?
    val_head = agent.components["behavior_heads"]["state_value"].output_layer
    assert val_head.weight.grad is not None, "Gradient failed to reach the Value head."

    # B. Verify Gradient Isolation (Stopped at Target)
    target_repr_backbone = target_agent.components["representation"].conv
    # In PyTorch, weights that haven't received gradients have .grad == None
    assert target_repr_backbone.weight.grad is None, \
        "CRITICAL: Gradients leaked into the Target/Momentum network during target generation."
        
    # C. Sanity Checks
    assert torch.isfinite(loss), "Loss became NaN or Inf."
    assert torch.isfinite(repr_backbone.weight.grad).all(), "Gradients became NaN or Inf in backbone."
