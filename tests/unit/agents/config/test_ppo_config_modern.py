import pytest
import torch
from configs.agents.ppo import PPOConfig
from configs.games.game import GameConfig
from agents.registries.ppo import build_ppo
from modules.models.agent_network import AgentNetwork
from agents.learner.losses.value import ClippedValueLoss, ValueLoss

# Mock Game Config
class MockGame:
    def __init__(self):
        self.num_actions = 4
        self.num_players = 1
        self.observation_shape = (8,)
        self.env_factory = lambda: None
        self.min_score = -1
        self.max_score = 1

@pytest.fixture
def game_config():
    return MockGame()

def test_ppo_config_parsing(game_config):
    config_dict = {
        "agent_type": "ppo",
        "learning_rate": 2.5e-4,
        "clip_param": 0.2,
        "clip_value_loss": False,
        "steps_per_epoch": 512,
        "action_selector": {"base": {"type": "categorical"}},
    }
    config = PPOConfig(config_dict, game_config)
    assert config.learning_rate == 2.5e-4
    assert config.clip_param == 0.2
    assert config.clip_value_loss is False

def test_ppo_registry_loss_selection(game_config):
    # Test Standard Value Loss (clip_value_loss=False)
    config_dict = {
        "agent_type": "ppo",
        "learning_rate": 2.5e-4,
        "clip_param": 0.2,
        "clip_value_loss": False,
        "steps_per_epoch": 512,
        "action_selector": {"base": {"type": "categorical"}},
        "policy_head": {"neck": {"type": "mlp", "widths": [64]}},
        "value_head": {"neck": {"type": "mlp", "widths": [64]}},
    }
    config = PPOConfig(config_dict, game_config)
    
    # Mock Agent Network
    network = AgentNetwork(
        input_shape=(8,),
        num_actions=4,
        head_fns={
            "policy_logits": lambda x: torch.randn(x.shape[0], 4),
            "state_value": lambda x: torch.randn(x.shape[0], 1),
        }
    )
    
    device = torch.device("cpu")
    components = build_ppo(config, network, device)
    loss_pipeline = components["loss_pipeline"]
    
    # Find value loss module
    value_loss = next(m for m in loss_pipeline.modules if m.name == "value_loss")
    assert isinstance(value_loss, ValueLoss)
    assert not isinstance(value_loss, ClippedValueLoss)

    # Test Clipped Value Loss (clip_value_loss=True)
    config_dict["clip_value_loss"] = True
    config = PPOConfig(config_dict, game_config)
    components = build_ppo(config, network, device)
    loss_pipeline = components["loss_pipeline"]
    value_loss = next(m for m in loss_pipeline.modules if m.name == "value_loss")
    assert isinstance(value_loss, ClippedValueLoss)

if __name__ == "__main__":
    pytest.main([__file__, "-o", "addopts=", "-p", "no:cacheprovider"])
