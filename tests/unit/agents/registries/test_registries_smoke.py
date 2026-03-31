import pytest
import torch
from typing import Any
from modules.agent_nets.factory import build_modular_agent_network
from configs.agents.muzero import MuZeroConfig
from configs.agents.rainbow_dqn import RainbowConfig
from configs.agents.supervised import SupervisedConfig
from configs.games.cartpole import CartPoleConfig
from configs.games.tictactoe import TicTacToeConfig

pytestmark = pytest.mark.unit

def test_muzero_registry_smoke():
    """Verifies that MuZero network can be built with default config."""
    config_dict = {
        "agent_type": "muzero",
        "architecture": {"type": "mlp"},
        "minibatch_size": 32,
        "action_selector": {"base": {"type": "muzero"}},
    }
    game_config = TicTacToeConfig()
    config = MuZeroConfig(config_dict, game_config)
    
    input_shape = (3, 3, 3) # Example obs shape
    num_actions = 9
    
    agent_network = build_modular_agent_network(
        config=config,
        input_shape=input_shape,
        num_actions=num_actions,
    )
    
    assert agent_network is not None
    assert "world_model" in agent_network.components
    assert "value_head" in agent_network.components
    assert "policy_head" in agent_network.components

def test_muzero_obs_inference_smoke():
    """Verifies that MuZero network can perform obs_inference (triggers initial_inference)."""
    config_dict = {
        "agent_type": "muzero",
        "architecture": {"type": "mlp"},
        "minibatch_size": 32,
        "action_selector": {"base": {"type": "muzero"}},
    }
    game_config = TicTacToeConfig()
    config = MuZeroConfig(config_dict, game_config)
    
    input_shape = (3, 3, 3) 
    num_actions = 9
    
    agent_network = build_modular_agent_network(
        config=config,
        input_shape=input_shape,
        num_actions=num_actions,
    )
    
    # Test with observation missing batch dim (common in actors)
    obs = torch.zeros(input_shape)
    with torch.inference_mode():
        output = agent_network.obs_inference(obs)
    
    assert output is not None
    assert output.value is not None

def test_rainbow_registry_smoke():
    """Verifies that Rainbow network can be built with default config."""
    config_dict = {
        "agent_type": "rainbow",
        "backbone": {"type": "mlp", "widths": [64]},
        "head": {"hidden_widths": [64], "output_strategy": {"type": "scalar"}},
        "minibatch_size": 32,
        "action_selector": {"base": {"type": "categorical"}},
    }
    game_config = CartPoleConfig()
    config = RainbowConfig(config_dict, game_config)
    
    input_shape = (4,)
    num_actions = 2
    
    agent_network = build_modular_agent_network(
        config=config,
        input_shape=input_shape,
        num_actions=num_actions,
    )
    
    assert agent_network is not None
    assert "q_head" in agent_network.components
    assert "feature_block" in agent_network.components

def test_supervised_registry_smoke():
    """Verifies that Supervised network can be built with default config."""
    config_dict = {
        "agent_type": "supervised",
        "sl_backbone": {"type": "mlp", "widths": [64]},
        "policy_head": {"neck": {"type": "identity"}, "output_strategy": {"type": "categorical", "num_classes": 2}},
        "sl_minibatch_size": 32,
        "sl_loss_function": "cross_entropy",
        "training_steps": 100,
    }
    config = SupervisedConfig(config_dict)
    
    input_shape = (4,)
    num_actions = 2
    
    agent_network = build_modular_agent_network(
        config=config,
        input_shape=input_shape,
        num_actions=num_actions,
    )
    
    assert agent_network is not None
    assert "policy_head" in agent_network.components
    assert "feature_block" in agent_network.components
