import torch
import torch.nn as nn
import pytest
from typing import Tuple, Dict, Any, Callable

# Project Imports
from modules.models.agent_network import AgentNetwork
from modules.models.world_model import WorldModel
from modules.heads.policy import PolicyHead
from modules.heads.value import ValueHead
from modules.heads.reward import RewardHead
from agents.learner.losses.representations import ClassificationRepresentation, DiscreteSupportRepresentation

pytestmark = pytest.mark.integration

class DummyBackbone(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], output_channels: int = 8):
        super().__init__()
        self.input_shape = input_shape
        # Simple projection to change channels if needed
        self.conv = nn.Conv2d(input_shape[0], output_channels, 3, padding=1)
        self.output_shape = (output_channels, *input_shape[1:])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

def test_muzero_network_unroll_shapes():
    """
    Ensure the Dynamics and Prediction networks can unroll K times without shape mismatch errors.
    
    Setup:
    - B=2 (Batch size)
    - K=3 (Unroll steps)
    - C, H, W = 3, 8, 8 (Input shape)
    - num_actions = 4
    
    Assertion:
    1. Pass observations through the Representation network. assert hidden_state.shape == expected.
    2. Loop K times, passing the hidden_state and actions[:, k] into the Dynamics network.
    3. Assert the final returned lists for policies, values, and rewards all have length K+1.
    """
    B, K = 2, 3
    C, H, W = 3, 8, 8
    num_actions = 4
    latent_channels = 8
    latent_dim = (latent_channels, H, W)
    
    # 1. Component Factories
    def repr_fn(input_shape: Tuple[int, ...]) -> nn.Module:
        return DummyBackbone(input_shape, output_channels=latent_channels)

    def world_model_fn(latent_dimensions: Tuple[int, ...], num_actions: int, num_players: int) -> nn.Module:
        def dynamics_fn(input_shape: Tuple[int, ...]) -> nn.Module:
            # Must return a module with .output_shape
            m = nn.Conv2d(input_shape[0], latent_dimensions[0], 3, padding=1)
            m.output_shape = latent_dimensions
            return m
        
        def reward_head_fn(input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
            return RewardHead(
                input_shape=input_shape,
                representation=DiscreteSupportRepresentation(vmin=-10, vmax=10, bins=21),
                name="reward_logits"
            )
            
        return WorldModel(
            latent_dimensions=latent_dimensions,
            num_actions=num_actions,
            num_players=num_players,
            dynamics_fn=dynamics_fn,
            env_head_fns={"reward_logits": reward_head_fn}
        )

    head_fns = {
        "policy_logits": lambda input_shape, num_actions, **kwargs: PolicyHead(
            input_shape=input_shape,
            representation=ClassificationRepresentation(num_actions),
            name="policy_logits"
        ),
        "state_value": lambda input_shape, **kwargs: ValueHead(
            input_shape=input_shape,
            representation=DiscreteSupportRepresentation(vmin=-10, vmax=10, bins=21),
            name="state_value"
        )
    }

    # 2. Instantiate AgentNetwork
    agent = AgentNetwork(
        input_shape=(C, H, W),
        num_actions=num_actions,
        representation_fn=repr_fn,
        world_model_fn=world_model_fn,
        head_fns=head_fns
    )
    agent.eval() # Set to eval mode for deterministic check
    
    # 3. Create dummy inputs
    obs = torch.randn(B, C, H, W)
    actions = torch.randint(0, num_actions, (B, K))
    
    # --- STEP 1: Representation (Root Prediction) ---
    root_out = agent.obs_inference(obs)
    
    # hidden_state is a dict containing 'dynamics' latent
    hidden_state = root_out.recurrent_state
    assert "dynamics" in hidden_state, "Recurrent state must contain 'dynamics' key for MuZero."
    assert hidden_state["dynamics"].shape == (B, *latent_dim), f"Expected {(B, *latent_dim)}, got {hidden_state['dynamics'].shape}"
    
    # --- STEP 2: Unroll Loop (K steps) ---
    policies = [root_out.policy] # Root policy
    values = [root_out.value]     # Root value
    rewards = [None]              # Root has no reward in standard MuZero
    
    current_state = hidden_state
    for k in range(K):
        # Progress Physics and Predict Heads for step k+1
        unroll_out = agent.hidden_state_inference(current_state, actions[:, k])
        
        policies.append(unroll_out.policy)
        values.append(unroll_out.value)
        rewards.append(unroll_out.reward)
        
        # Update current state for next iteration
        current_state = unroll_out.recurrent_state
        
        # Shape Invariant Check: Dynamics must preserve latent shape
        assert current_state["dynamics"].shape == (B, *latent_dim)
        
    # --- STEP 3: Final Assertions ---
    # Expected lengths: 1 root + K unroll steps = K+1
    assert len(policies) == K + 1, f"Expected {K+1} policies, got {len(policies)}"
    assert len(values) == K + 1, f"Expected {K+1} values, got {len(values)}"
    assert len(rewards) == K + 1, f"Expected {K+1} rewards, got {len(rewards)}"

    # Check that predictions are valid distributions/tensors
    for i, (p, v) in enumerate(zip(policies, values)):
        # Policies should be Distribution objects yielding correct log_probs/entropy shapes
        assert isinstance(p, torch.distributions.Distribution), f"Step {i}: Policy must be a Distribution object"
        assert p.batch_shape == (B,), f"Step {i}: Batch shape mismatch. Expected {(B,)}, got {p.batch_shape}"
        
        # Values should be tensors of shape (B,) representing the expected value
        assert torch.is_tensor(v), f"Step {i}: Value must be a Tensor"
        assert v.shape == (B,), f"Step {i}: Value shape mismatch. Expected {(B,)}, got {v.shape}"

    # Rewards start from index 1 (transitions)
    for i, r in enumerate(rewards[1:], 1):
        assert torch.is_tensor(r), f"Step {i}: Reward must be a Tensor"
        assert r.shape == (B,), f"Step {i}: Reward shape mismatch. Expected {(B,)}, got {r.shape}"
