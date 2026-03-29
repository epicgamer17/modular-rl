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
from agents.learner.losses.representations import (
    ClassificationRepresentation,
    DiscreteSupportRepresentation,
)

pytestmark = pytest.mark.integration


class SimpleTestBackbone(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], output_channels: int = 8):
        super().__init__()
        self.input_shape = input_shape
        self.conv = nn.Conv2d(input_shape[0], output_channels, 3, padding=1)
        self.output_shape = (output_channels, *input_shape[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def build_muzero_network(C, H, W, num_actions, latent_channels=8):
    def repr_fn(input_shape):
        return SimpleTestBackbone(input_shape, output_channels=latent_channels)

    def world_model_fn(latent_dimensions, num_actions, num_players):
        def dynamics_fn(input_shape):
            m = nn.Conv2d(input_shape[0], latent_dimensions[0], 3, padding=1)
            m.output_shape = latent_dimensions
            return m

        def reward_head_fn(input_shape, **kwargs):
            return RewardHead(
                input_shape=input_shape,
                representation=DiscreteSupportRepresentation(vmin=-1, vmax=1, bins=3),
                name="reward_logits",
            )

        return WorldModel(
            latent_dimensions=latent_dimensions,
            num_actions=num_actions,
            num_players=num_players,
            dynamics_fn=dynamics_fn,
            env_head_fns={"reward_logits": reward_head_fn},
        )

    head_fns = {
        "policy_logits": lambda input_shape, num_actions, **kwargs: PolicyHead(
            input_shape=input_shape,
            representation=ClassificationRepresentation(num_actions),
            name="policy_logits",
        ),
        "state_value": lambda input_shape, **kwargs: ValueHead(
            input_shape=input_shape,
            representation=DiscreteSupportRepresentation(vmin=-1, vmax=1, bins=3),
            name="state_value",
        ),
    }

    return AgentNetwork(
        input_shape=(C, H, W),
        num_actions=num_actions,
        representation_fn=repr_fn,
        world_model_fn=world_model_fn,
        head_fns=head_fns,
    )


def test_muzero_loss_pipeline_gradient_flow():
    """
    Tier 1 Integration Test: Ultimate Gradient Flow Check.
    - Pass a synthetic batch through K=5 unrolled steps.
    - Compute the combined loss and call .backward().
    - Assert: representation_net weight grad is not None.
    - Assert: dynamics_net weight grad is not None.
    - Assert: prediction_net weight grad is not None.
    - Assert: Target network gradients are strictly None.
    """
    B, K = 2, 5
    C, H, W = 3, 4, 4
    num_actions = 4

    # 1. Setup Online and Target Networks
    agent = build_muzero_network(C, H, W, num_actions)
    target_agent = build_muzero_network(C, H, W, num_actions)

    agent.zero_grad(set_to_none=True)
    target_agent.zero_grad(set_to_none=True)

    # 2. Synthetic Batch
    obs = torch.randn(B, C, H, W)
    actions = torch.randint(0, num_actions, (B, K))

    # 3. Unroll Online Network (K=5)
    # This involves 1 representation step + 5 dynamics steps
    root_out = agent.obs_inference(obs)

    current_state = root_out.recurrent_state
    all_values = [root_out.value]
    all_policies = [root_out.policy]
    all_rewards = []

    for k in range(K):
        # Step through dynamics
        unroll_out = agent.hidden_state_inference(current_state, actions[:, k])
        all_values.append(unroll_out.value)
        all_policies.append(unroll_out.policy)
        all_rewards.append(unroll_out.reward)
        current_state = unroll_out.recurrent_state

    # 4. Compute Loss
    # We create target values from the target agent to simulate fixed targets
    with torch.no_grad():
        target_out = target_agent.obs_inference(obs)
        target_v = target_out.value.detach()
        # Mocking unrolled targets as the same for simplicity
        target_policy_logits = torch.zeros(B, num_actions)
        target_policy_logits[:, 0] = 1.0  # Target action 0
        target_reward = torch.zeros(B)

    # Use a dummy loss on the LAST prediction step (K=5) to ensure full flow
    # Loss = MSE(value_K, target_v) + CE(policy_K, target_p) + MSE(reward_K, target_r)
    loss_v = torch.nn.functional.mse_loss(all_values[-1], target_v)

    # Policy loss for last step
    policy_last = all_policies[-1]  # This is a Distribution
    # Accessing logits from distribution
    loss_p = -policy_last.log_prob(torch.zeros(B, dtype=torch.long)).mean()

    # Reward loss for last step
    loss_r = torch.nn.functional.mse_loss(all_rewards[-1], target_reward)

    total_loss = loss_v + loss_p + loss_r
    total_loss.backward()

    # 5. Assertions
    # A. Representation Network
    repr_weight = agent.components["representation"].conv.weight
    assert repr_weight.grad is not None, "Representation gradient is None"
    assert not torch.all(repr_weight.grad == 0), "Representation gradient is all zeros"

    # B. Dynamics Network
    # The dynamics model is inside world_model
    dynamics_weight = agent.components["world_model"].dynamics_pipeline.dynamics.weight
    assert dynamics_weight.grad is not None, "Dynamics gradient is None"
    assert not torch.all(dynamics_weight.grad == 0), "Dynamics gradient is all zeros"

    # C. Prediction Heads (Policy and Value)
    policy_weight = agent.components["behavior_heads"][
        "policy_logits"
    ].output_layer.weight
    value_weight = agent.components["behavior_heads"]["state_value"].output_layer.weight
    assert policy_weight.grad is not None, "Policy head gradient is None"
    assert value_weight.grad is not None, "Value head gradient is None"

    # D. Reward Head
    reward_weight = (
        agent.components["world_model"].heads["reward_logits"].output_layer.weight
    )
    assert reward_weight.grad is not None, "Reward head gradient is None"

    # E. Target Network (ISOLATION)
    target_repr_weight = target_agent.components["representation"].conv.weight
    assert (
        target_repr_weight.grad is None
    ), "Target network gradient is not None (Leakage!)"

    print(f"Passed Muzero Gradient Flow check with K={K} unroll steps!")
