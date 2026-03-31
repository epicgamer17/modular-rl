import pytest
import torch
import numpy as np
from agents.factories.learner import build_universal_learner
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
    
    network = net_factory(ppo_config, obs_shape, num_actions)
    learner = build_universal_learner(ppo_config, network, device)
    
    # Batch as returned by PPOBatchProcessor
    batch_size = 4
    batch = {
        "observations": torch.randn((batch_size, *obs_shape)),
        "actions": torch.randint(0, num_actions, (batch_size,)),
        "values": torch.randn((batch_size,)),
        "advantages": torch.randn((batch_size,)),
        "returns": torch.randn((batch_size,)),
        "log_prob": torch.randn((batch_size,)),
        "legal_moves_masks": torch.ones((batch_size, num_actions), dtype=torch.bool),
    }
    
    # This call failed with KeyError: 'values' before the fix
    result = learner.compute_step_result(batch)
    
    assert "policy_loss" in result.loss_dict
    assert "value_loss" in result.loss_dict
    assert result.targets["values"].shape == (batch_size, 1) # Padded to T=1
    assert result.targets["returns"].shape == (batch_size, 1)

def test_muzero_loss_contract(muzero_config, cartpole_game_config, net_factory):
    """
    Verifies that the MuZero TargetBuilder and LossPipeline have a consistent contract across unrolled steps.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    obs_shape = (4,)
    num_actions = cartpole_game_config.num_actions
    unroll_steps = muzero_config.unroll_steps
    T = unroll_steps + 1
    
    # We need a real world model for latent consistency if enabled
    muzero_config.consistency_loss_factor = 0 # Simplify for this contract test
    network = net_factory(muzero_config, obs_shape, num_actions)
    learner = build_universal_learner(muzero_config, network, device)
    
    # Unrolled batch as returned by NStepUnrollProcessor
    batch_size = 2
    batch = {
        "observations": torch.randn((batch_size, *obs_shape)),
        "actions": torch.zeros((batch_size, T), dtype=torch.float32), 
        "values": torch.randn((batch_size, T)),
        "rewards": torch.randn((batch_size, T)),
        "policies": torch.randn((batch_size, T, num_actions)),
        # Helper keys for sequence building
        "is_same_game": torch.ones((batch_size, T), dtype=torch.bool),
        "dones": torch.zeros((batch_size, T), dtype=torch.bool),
    }
    
    result = learner.compute_step_result(batch)
    
    assert "value_loss" in result.loss_dict
    assert "policy_loss" in result.loss_dict
    assert "reward_loss" in result.loss_dict
    # Check that targets were correctly padded/formatted for MuZero sequence loss
    assert result.targets["values"].shape == (batch_size, T)
    assert result.targets["policies"].shape == (batch_size, T, num_actions)

def test_rainbow_loss_contract(rainbow_config, cartpole_game_config, net_factory):
    """
    Verifies that the Rainbow TargetBuilder (TD/Distributional) and LossPipeline have a consistent contract.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    obs_shape = (4,)
    num_actions = cartpole_game_config.num_actions
    
    # Create target network (Rainbow requirement)
    target_network = net_factory(rainbow_config, obs_shape, num_actions)
    network = net_factory(rainbow_config, obs_shape, num_actions)
    
    learner = build_universal_learner(
        rainbow_config, network, device, target_agent_network=target_network
    )
    
    batch_size = 4
    batch = {
        "observations": torch.randn((batch_size, *obs_shape)),
        "actions": torch.randint(0, num_actions, (batch_size,)),
        "rewards": torch.randn((batch_size,)),
        "next_observations": torch.randn((batch_size, *obs_shape)),
        "dones": torch.zeros((batch_size,), dtype=torch.bool),
        "next_legal_moves_masks": torch.ones((batch_size, num_actions), dtype=torch.bool),
    }
    
    result = learner.compute_step_result(batch)
    
    assert "QBootstrappingLoss" in result.loss_dict
    # TemporalDifferenceBuilder produces q_values
    assert "q_values" in result.targets
    assert result.targets["q_values"].shape == (batch_size, 1) # Padded T=1
