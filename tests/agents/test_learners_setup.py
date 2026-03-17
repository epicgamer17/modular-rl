import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from agents.learners.ppo_learner import PPOLearner
from agents.learners.muzero_learner import MuZeroLearner
from agents.learners.base import UniversalLearner
from agents.learners.target_builders import DQNTargetBuilder
from configs.agents.ppo import PPOConfig
from configs.agents.rainbow_dqn import RainbowConfig
from configs.agents.muzero import MuZeroConfig
from configs.agents.supervised import SupervisedConfig
from configs.games.cartpole import CartPoleConfig
from losses.losses import C51Loss, ImitationLoss, LossPipeline, StandardDQNLoss
from modules.agent_nets.modular import ModularAgentNetwork
from modules.utils import get_lr_scheduler

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
    obs_dim = (4,)
    num_actions = 10

    agent_network = ModularAgentNetwork(config=config, input_shape=obs_dim, num_actions=num_actions).to(device)
    target_network = ModularAgentNetwork(config=config, input_shape=obs_dim, num_actions=num_actions).to(device)

    optimizer = torch.optim.Adam(
        params=agent_network.parameters(),
        lr=config.learning_rate,
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    lr_scheduler = get_lr_scheduler(optimizer, config)

    target_builder = DQNTargetBuilder(
        device=device,
        target_network=target_network,
        gamma=config.discount_factor,
        n_step=config.n_step,
        use_c51=config.atom_size > 1,
        v_min=getattr(config, "v_min", None),
        v_max=getattr(config, "v_max", None),
        atom_size=getattr(config, "atom_size", 1),
        bootstrap_on_truncated=getattr(config, "bootstrap_on_truncated", False),
    )
    td_loss = StandardDQNLoss(config=config, device=device)
    loss_pipeline = LossPipeline([td_loss])
    loss_pipeline.validate_dependencies(
        network_output_keys={"q_values"},
        target_keys={"q_values", "actions"},
    )

    learner = UniversalLearner(
        config=config,
        agent_network=agent_network,
        device=device,
        num_actions=num_actions,
        observation_dimensions=obs_dim,
        observation_dtype=torch.float32,
        target_builder=target_builder,
        loss_pipeline=loss_pipeline,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    learner.target_agent_network = target_network

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
    config = SupervisedConfig(config_dict)
    obs_dim = (4,)
    num_actions = 10

    agent_network = ModularAgentNetwork(config=config, input_shape=obs_dim, num_actions=num_actions).to(device)
    optimizer = torch.optim.Adam(
        params=agent_network.parameters(),
        lr=config.learning_rate,
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    lr_scheduler = get_lr_scheduler(optimizer, config)

    loss_pipeline = LossPipeline([ImitationLoss(config, device, num_actions)])
    loss_pipeline.validate_dependencies(
        network_output_keys={"policies"},
        target_keys={"target_policies"},
    )

    learner = UniversalLearner(
        config=config,
        agent_network=agent_network,
        device=device,
        num_actions=num_actions,
        observation_dimensions=obs_dim,
        observation_dtype=torch.float32,
        target_builder=None,
        loss_pipeline=loss_pipeline,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    assert hasattr(learner, "loss_pipeline")
    assert len(learner.loss_pipeline.modules) > 0
