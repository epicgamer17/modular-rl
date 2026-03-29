import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from agents.registries.muzero import build_muzero
from agents.registries.ppo import build_ppo
from agents.registries.rainbow import build_rainbow
from configs.agents.muzero import MuZeroConfig
from configs.agents.ppo import PPOConfig
from configs.agents.rainbow_dqn import RainbowConfig
from configs.games.game import GameConfig
from agents.learner.losses.representations import ScalarRepresentation, ClassificationRepresentation

pytestmark = pytest.mark.unit

class MockGame(GameConfig):
    def __init__(self):
        self.observation_shape = (8,)
        self.num_actions = 4
        self.num_players = 1
        self.min_score = 0
        self.max_score = 1
        self.is_discrete = True
        self.env_factory = lambda: None

class SimpleNet(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super().__init__()
        self.fc = nn.Linear(obs_dim, num_actions)
        self.num_actions = num_actions
        self.input_shape = (obs_dim,)
        # Mock components for registries that look inside
        self.components = {
            "behavior_heads": {
                "state_value": nn.Module(),
                "policy_logits": nn.Module(),
                "q_logits": nn.Module(),
            },
            "world_model": nn.Module()
        }
        # Add representations to mock components
        self.components["behavior_heads"]["state_value"].representation = ScalarRepresentation()
        self.components["behavior_heads"]["policy_logits"].representation = ClassificationRepresentation(num_actions)
        self.components["behavior_heads"]["q_logits"].representation = ScalarRepresentation()
        
        # Add actual parameters to heads to satisfy PPO/Rainbow registries
        self.components["behavior_heads"]["state_value"].register_parameter("p", nn.Parameter(torch.randn(1)))
        self.components["behavior_heads"]["policy_logits"].register_parameter("p", nn.Parameter(torch.randn(1)))
        self.components["behavior_heads"]["q_logits"].register_parameter("p", nn.Parameter(torch.randn(1)))
        
        # For MuZero
        self.components["world_model"].heads = {
            "reward_logits": nn.Module(),
            "to_play_logits": nn.Module(),
        }
        self.components["world_model"].heads["reward_logits"].representation = ScalarRepresentation()
        self.components["world_model"].heads["to_play_logits"].representation = ClassificationRepresentation(1)
        self.components["world_model"].heads["reward_logits"].register_parameter("p", nn.Parameter(torch.randn(1)))
        self.components["world_model"].heads["to_play_logits"].register_parameter("p", nn.Parameter(torch.randn(1)))

    def parameters(self, recurse=True):
        return self.fc.parameters(recurse)

def test_muzero_adam_epsilon_config():
    """Verify that MuZero registry respects adam_epsilon configuration."""
    game = MockGame()
    config_dict = {
        "adam_epsilon": 1e-5,
        "learning_rate": 1e-3,
        "agent_type": "muzero",
        "action_selector": {"base": {"type": "categorical"}},
    }
    config = MuZeroConfig(config_dict, game)
    net = SimpleNet(8, 4)
    device = torch.device("cpu")
    
    components = build_muzero(config, net, device)
    optimizer = components["optimizers"]["default"]
    
    assert optimizer.param_groups[0]["eps"] == 1e-5, f"Expected eps=1e-5, got {optimizer.param_groups[0]['eps']}"

def test_ppo_adam_epsilon_config():
    """Verify that PPO registry respects adam_epsilon configuration for both actor and critic."""
    game = MockGame()
    config_dict = {
        "adam_epsilon": 1e-5,
        "learning_rate": 1e-3,
        "agent_type": "ppo",
        "steps_per_epoch": 128,
        "action_selector": {"base": {"type": "categorical"}},
    }
    config = PPOConfig(config_dict, game)
    net = SimpleNet(8, 4)
    device = torch.device("cpu")
    
    components = build_ppo(config, net, device)
    
    assert components["optimizers"]["policy"].param_groups[0]["eps"] == 1e-5
    assert components["optimizers"]["value"].param_groups[0]["eps"] == 1e-5

def test_rainbow_adam_epsilon_config():
    """Verify that Rainbow registry respects adam_epsilon configuration."""
    game = MockGame()
    config_dict = {
        "adam_epsilon": 1e-5,
        "learning_rate": 1e-3,
        "agent_type": "rainbow",
        "action_selector": {"base": {"type": "categorical"}},
    }
    config = RainbowConfig(config_dict, game)
    net = SimpleNet(8, 4)
    device = torch.device("cpu")
    
    target_net = SimpleNet(8, 4)
    components = build_rainbow(config, net, device, target_agent_network=target_net)
    optimizer = components["optimizers"]["default"]
    
    assert optimizer.param_groups[0]["eps"] == 1e-5

def test_adam_epsilon_default():
    """Verify that the default adam_epsilon is 1e-8."""
    game = MockGame()
    config_dict = {
        "learning_rate": 1e-3,
        "agent_type": "muzero",
        "action_selector": {"base": {"type": "categorical"}},
    }
    config = MuZeroConfig(config_dict, game)
    net = SimpleNet(8, 4)
    device = torch.device("cpu")
    
    components = build_muzero(config, net, device)
    optimizer = components["optimizers"]["default"]
    
    assert optimizer.param_groups[0]["eps"] == 1e-8
