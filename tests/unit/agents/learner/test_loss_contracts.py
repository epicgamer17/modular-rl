import pytest
import torch
import numpy as np
from agents.factories.learner import build_universal_learner
from agents.factories.model import build_agent_network
from configs.agents.rainbow_dqn import RainbowConfig
from agents.learner.losses import LossPipeline

pytestmark = pytest.mark.unit


def test_ppo_loss_contract(ppo_config, cartpole_game_config, net_factory):
    """
    Verifies that the PPO TargetBuilder and LossPipeline have a consistent contract.
    Specifically checks that 'values' (old values) are present for ClippedValueLoss.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    obs_shape = (4,)
    num_actions = cartpole_game_config.num_actions

    network = build_agent_network(ppo_config, obs_shape, num_actions)
    learner = build_universal_learner(ppo_config, network, device)

    # Batch as returned by PPOBatchProcessor
    batch_size = ppo_config.minibatch_size
    batch = {
        "observations": torch.randn((batch_size, *obs_shape)),
        "actions": torch.randint(0, num_actions, (batch_size,)),
        # PPO requires 'values' and 'returns' (etc) from the buffer
        "values": torch.randn((batch_size, 1)),  # PPO targets are single-step
        "returns": torch.randn((batch_size, 1)),
        "log_prob": torch.randn((batch_size, 1)),
        "advantages": torch.randn((batch_size, 1)),
    }

    result = learner.compute_step_result(batch)
    
    # 1. Verify KeyError is gone (targets built and run)
    assert "values" in result.targets
    assert result.targets["values"].shape == (batch_size, 1)
    
    # 2. Verify all loss components reported metrics in loss_dict
    assert "policy_loss" in result.loss_dict
    assert "value_loss" in result.loss_dict

def test_muzero_loss_contract(muzero_config, cartpole_game_config):
    """
    Verifies that the MuZero TargetBuilder (Unrolled) and LossPipeline have a consistent contract.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    obs_shape = (4,)
    num_actions = cartpole_game_config.num_actions
    
    # Ensure MuZero is correctly configured for sequence unroll
    muzero_config.consistency_loss_factor = 0 
    muzero_config.unroll_steps = 2 # Small unroll for test
    
    network = build_agent_network(muzero_config, obs_shape, num_actions)
    learner = build_universal_learner(muzero_config, network, device)
    
    # Unrolled batch as returned by NStepUnrollProcessor
    batch_size = muzero_config.minibatch_size
    T = muzero_config.unroll_steps
    batch = {
        "observations": torch.randn((batch_size, T + 1, *obs_shape)),
        "actions": torch.randint(0, num_actions, (batch_size, T)), 
        "values": torch.randn((batch_size, T + 1)),
        "rewards": torch.randn((batch_size, T)),
        "dones": torch.zeros((batch_size, T), dtype=torch.bool),
        "is_same_game": torch.ones((batch_size, T + 1), dtype=torch.bool),
    }

    result = learner.compute_step_result(batch)
    
    # Verify contract
    assert "values" in result.targets
    assert "rewards" in result.targets
    assert result.targets["values"].shape == (batch_size, T + 1)
    # T+1 because SequencePadder pads the T rewards to T+1
    assert result.targets["rewards"].shape == (batch_size, T + 1)

def test_rainbow_loss_contract(rainbow_config, cartpole_game_config, make_rainbow_config_dict):
    """
    Verifies that the Rainbow TargetBuilder (TD/Distributional) and LossPipeline have a consistent contract.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    obs_shape = (4,)
    num_actions = cartpole_game_config.num_actions
    
    # Explicitly ensure Rainbow is categorical for this test
    c_dict = make_rainbow_config_dict(atom_size=51)
    rainbow_config = RainbowConfig(c_dict, cartpole_game_config)
    
    # Create target network (Rainbow requirement)
    target_network = build_agent_network(rainbow_config, obs_shape, num_actions)
    network = build_agent_network(rainbow_config, obs_shape, num_actions)
    
    learner = build_universal_learner(
        rainbow_config, network, device, target_agent_network=target_network
    )
    
    batch_size = rainbow_config.minibatch_size
    batch = {
        "observations": torch.randn((batch_size, *obs_shape)),
        "actions": torch.randint(0, num_actions, (batch_size,)),
        "rewards": torch.randn((batch_size,)),
        "next_observations": torch.randn((batch_size, *obs_shape)),
        "dones": torch.zeros((batch_size,), dtype=torch.bool),
        "next_legal_moves_masks": torch.ones((batch_size, num_actions), dtype=torch.bool),
    }
    
    result = learner.compute_step_result(batch)
    
    # Verify contract
    # DistributionalTargetBuilder produces q_logits
    assert "q_logits" in result.targets
    assert "QBootstrappingLoss" in result.loss_dict
    # Verify time dimension was added by SingleStepFormatter
    assert result.targets["q_logits"].shape == (batch_size, 1, 51)
