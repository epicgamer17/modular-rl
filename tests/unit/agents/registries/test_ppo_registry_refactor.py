import pytest
import torch
from typing import Any
from agents.registries.ppo import build_ppo, build_ppo_loss_pipeline
from modules.agent_nets.modular import ModularAgentNetwork

pytestmark = pytest.mark.unit

def test_build_ppo_loss_pipeline_no_config(ppo_config):
    """Verifies that build_ppo_loss_pipeline can be called with explicit arguments."""
    # 1. Setup mock/dummy agent network
    input_shape = (4,)
    num_actions = 2
    agent_network = ModularAgentNetwork(
        config=ppo_config,
        input_shape=input_shape,
        num_actions=num_actions,
    )
    device = torch.device("cpu")

    # 2. Call refactored function
    loss_pipeline = build_ppo_loss_pipeline(
        agent_network=agent_network,
        device=device,
        clip_param=0.2,
        entropy_coefficient=0.01,
        critic_coefficient=0.5,
        minibatch_size=32,
        num_actions=num_actions,
    )

    # 3. Assertions
    assert loss_pipeline is not None
    assert len(loss_pipeline.modules) == 2
    assert loss_pipeline.shape_validator.B == 32
    assert loss_pipeline.shape_validator.num_actions == num_actions
    assert loss_pipeline.shape_validator.T == 1 # PPO is single-step

def test_build_ppo_registry(ppo_config):
    """Verifies that build_ppo still works and produces a valid loss pipeline."""
    input_shape = (4,)
    num_actions = 2
    agent_network = ModularAgentNetwork(
        config=ppo_config,
        input_shape=input_shape,
        num_actions=num_actions,
    )
    device = torch.device("cpu")

    # Call registry function
    agent_bundle = build_ppo(ppo_config, agent_network, device)

    # Assertions
    assert "loss_pipeline" in agent_bundle
    loss_pipeline = agent_bundle["loss_pipeline"]
    assert len(loss_pipeline.modules) == 2
    # Verify values were correctly extracted from config
    assert loss_pipeline.shape_validator.B == ppo_config.minibatch_size
    assert loss_pipeline.shape_validator.num_actions == ppo_config.game.num_actions
