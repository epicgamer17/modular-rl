import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from agents.learners.ppo_learner import PPOLearner
from agents.learners.rainbow_learner import RainbowLearner
from agents.learners.muzero_learner import MuZeroLearner
from agents.learners.imitation_learner import ImitationLearner
from configs.agents.ppo import PPOConfig
from configs.agents.rainbow_dqn import RainbowConfig
from configs.agents.muzero import MuZeroConfig
from configs.agents.supervised import SupervisedConfig
from configs.games.cartpole import CartPoleConfig

pytestmark = pytest.mark.unit


class SimpleNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.backbone = nn.Linear(obs_dim, 16)
        self.policy_head = nn.Linear(16, action_dim)
        self.value_head = nn.Linear(16, 21)
        self.reward_head = nn.Linear(16, 21)
        self.to_play_head = nn.Linear(16, 2)

        self.components = {
            "policy_head": self.policy_head,
            "value_head": self.value_head,
            "reward_head": self.reward_head,
            "to_play_head": self.to_play_head,
        }
        # Add strategy attribute to heads if needed by learners
        self.policy_head.strategy = None
        self.value_head.strategy = None
        self.training_action_selector = MagicMock()
        self.target_action_selector = MagicMock()
        self.project = nn.Linear(16, 16)

    def learner_inference(self, *args, **kwargs):
        out = MagicMock()
        return out

    def obs_inference(self, obs):
        out = MagicMock()
        return out

    def reset_noise(self):
        pass


@pytest.fixture
def device():
    return torch.device("cpu")


def test_ppo_learner_setup(make_ppo_config_dict, cartpole_game_config, device):
    config_dict = make_ppo_config_dict()
    config = PPOConfig(config_dict, cartpole_game_config)
    agent_network = SimpleNet(4, 10)

    learner = PPOLearner(
        config=config,
        agent_network=agent_network,
        device=device,
        num_actions=10,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
    )

    assert hasattr(learner, "policy_pipeline")
    assert hasattr(learner, "value_pipeline")
    assert len(learner.policy_pipeline.modules) > 0
    assert len(learner.value_pipeline.modules) > 0


def test_rainbow_learner_setup(make_rainbow_config_dict, cartpole_game_config, device):
    config_dict = make_rainbow_config_dict(atom_size=1)
    config = RainbowConfig(config_dict, cartpole_game_config)
    agent_network = SimpleNet(4, 10)
    target_network = SimpleNet(4, 10)

    learner = RainbowLearner(
        config=config,
        agent_network=agent_network,
        target_agent_network=target_network,
        device=device,
        num_actions=10,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
    )

    assert hasattr(learner, "loss_pipeline")
    assert len(learner.loss_pipeline.modules) > 0


def test_muzero_learner_setup(make_muzero_config_dict, cartpole_game_config, device):
    config_dict = make_muzero_config_dict()
    config = MuZeroConfig(config_dict, cartpole_game_config)
    agent_network = SimpleNet(4, 10)

    learner = MuZeroLearner(
        config=config,
        agent_network=agent_network,
        device=device,
        num_actions=10,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        player_id_mapping={0: 0, 1: 1},
    )

    assert hasattr(learner, "loss_pipeline")
    assert len(learner.loss_pipeline.modules) > 0


def test_imitation_learner_setup(make_supervised_config_dict, device):
    config_dict = make_supervised_config_dict()
    config_dict["replay_buffer_size"] = 100
    config_dict["minibatch_size"] = 2
    config_dict["optimizer"] = torch.optim.Adam
    config_dict["learning_rate"] = 1e-3
    config_dict["adam_epsilon"] = 1e-8
    config_dict["weight_decay"] = 0.0
    config_dict["sl_lr_schedule"] = {"type": "constant", "initial": 1e-3}
    config_dict["sl_training_iterations"] = 1

    config = SupervisedConfig(config_dict)
    agent_network = SimpleNet(4, 10)

    learner = ImitationLearner(
        config=config,
        agent_network=agent_network,
        device=device,
        num_actions=10,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
    )

    assert hasattr(learner, "loss_pipeline")
    assert len(learner.loss_pipeline.modules) > 0
