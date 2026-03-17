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
from modules.world_models.inference_output import LearningOutput, MuZeroNetworkState

pytestmark = pytest.mark.integration


class SimpleNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.backbone = nn.Linear(obs_dim, 16)
        self.policy_head = nn.Linear(16, action_dim)
        self.value_head = nn.Linear(16, 21)  # 21 atoms
        self.reward_head = nn.Linear(16, 21)
        self.to_play_head = nn.Linear(16, 2)
        self.dynamics = nn.Linear(16 + action_dim, 16)

        self.components = nn.ModuleDict(
            {
                "policy_head": self.policy_head,
                "value_head": self.value_head,
                "reward_head": self.reward_head,
                "to_play_head": self.to_play_head,
                "feature_block": self.backbone,
            }
        )
        self.training_action_selector = MagicMock()
        self.target_action_selector = MagicMock()
        self._project = nn.Linear(16, 16)

    def project(self, x, grad=True):
        return self._project(x)

    def learner_inference(self, batch):
        B = batch["observations"].shape[0]
        # Detect if we are unrolling (MuZero) or single-step (DQN/PPO)
        K_plus_1 = (
            4 if "unroll_observations" in batch or "unroll_actions" in batch else 1
        )
        # MuZero uses 21-atom distributional values/rewards; scalar otherwise
        num_atoms = 21 if K_plus_1 > 1 else 1

        return LearningOutput(
            policies=torch.randn((B, K_plus_1, 10), requires_grad=True),
            values=torch.randn((B, K_plus_1, num_atoms), requires_grad=True),
            q_values=torch.randn((B, K_plus_1, 10), requires_grad=True),
            q_logits=torch.randn((B, K_plus_1, 10, 21), requires_grad=True),
            rewards=torch.randn((B, K_plus_1, num_atoms), requires_grad=True),
            to_plays=torch.randn((B, K_plus_1, 2), requires_grad=True),
            latents=torch.randn((B, K_plus_1, 16), requires_grad=True),
        )

    def obs_inference(self, obs):
        out = MagicMock()
        out.policy.logits = torch.randn((obs.shape[0], 10), requires_grad=True)
        out.value.logits = torch.randn((obs.shape[0], 21), requires_grad=True)
        out.network_state = MuZeroNetworkState(
            dynamics=torch.randn((obs.shape[0], 16))
        )
        return out

    def reset_noise(self):
        pass


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def ppo_setup(make_ppo_config_dict, cartpole_game_config, device):
    config_dict = make_ppo_config_dict()
    config_dict["support_range"] = None
    config_dict["v_min"] = -10
    config_dict["v_max"] = 10
    config_dict["atom_size"] = 1
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
    return learner


def test_ppo_learner_loss_integration(ppo_setup):
    learner = ppo_setup

    batch = {"observations": torch.randn((2, 4))}
    actions = torch.randint(0, 10, (2,))
    old_log_probs = torch.randn((2,))
    advantages = torch.randn((2,))
    returns = torch.randn((2,))

    losses = learner.compute_loss(
        batch=batch,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
        returns=returns,
    )

    assert "policy_loss" in losses
    assert "value_loss" in losses
    assert isinstance(losses["policy_loss"], torch.Tensor)
    assert isinstance(losses["value_loss"], torch.Tensor)


@pytest.fixture
def rainbow_setup(make_rainbow_config_dict, cartpole_game_config, device):
    config_dict = make_rainbow_config_dict(atom_size=1)
    config_dict["batch_size"] = 2
    config_dict["loss_function"] = torch.nn.functional.mse_loss
    config_dict["discount_factor"] = 0.99
    config_dict["n_step"] = 1
    config_dict["bootstrap_on_truncated"] = False

    config = RainbowConfig(config_dict, cartpole_game_config)
    agent_network = SimpleNet(4, 10)
    target_network = SimpleNet(4, 10)

    target_builder = DQNTargetBuilder(config, device, target_network)
    loss_pipeline = LossPipeline([StandardDQNLoss(config=config, device=device)])
    loss_pipeline.validate_dependencies(
        network_output_keys={"q_values"},
        target_keys={"q_values", "actions"},
    )

    learner = UniversalLearner(
        config=config,
        agent_network=agent_network,
        device=device,
        num_actions=10,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=target_builder,
        loss_pipeline=loss_pipeline,
        optimizer=MagicMock(),
    )
    learner.target_agent_network = target_network
    return learner


def test_rainbow_learner_loss_integration(rainbow_setup):
    learner = rainbow_setup

    batch = {
        "observations": torch.randn((2, 4)),
        "next_observations": torch.randn((2, 4)),
        "actions": torch.randint(0, 10, (2,)),
        "rewards": torch.randn((2,)),
        "dones": torch.zeros((2,)).bool(),
        "weights": torch.ones((2,)),
    }

    step_result = learner.compute_step_result(batch)
    loss = step_result.loss
    priorities = step_result.priorities

    assert isinstance(loss, torch.Tensor)
    assert priorities.shape == (2,)


@pytest.fixture
def muzero_setup(make_muzero_config_dict, cartpole_game_config, device):
    config_dict = make_muzero_config_dict()
    config_dict["support_range"] = 10
    config_dict["v_min"] = -10
    config_dict["v_max"] = 10
    config_dict["atom_size"] = 21
    config_dict["consistency_loss_factor"] = 1.0
    config_dict["stochastic"] = False
    config_dict["bootstrap_on_truncated"] = False
    config_dict["discount_factor"] = 0.99

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
    return learner


def test_muzero_learner_loss_integration(muzero_setup):
    learner = muzero_setup

    batch = {
        "observations": torch.randn((2, 4)),
        "unroll_observations": torch.randn((2, 3, 4)),
        "actions": torch.randint(0, 10, (2, 3)),
        "rewards": torch.randn((2, 3)),
        "values": torch.randn((2, 4)),
        "policies": torch.randn((2, 4, 10)).softmax(dim=-1),
        "weights": torch.ones((2,)),
        "is_same_game": torch.ones((2, 4)).bool(),
        "has_valid_obs_mask": torch.ones((2, 4)).bool(),
        "has_valid_action_mask": torch.ones((2, 4)).bool(),
        "to_plays": torch.randint(0, 2, (2, 4)),
    }

    step_result = learner.compute_step_result(batch)

    assert isinstance(step_result.loss, torch.Tensor)
    assert step_result.priorities.shape == (2,)


@pytest.fixture
def imitation_setup(make_supervised_config_dict, device):
    config_dict = make_supervised_config_dict(
        sl_minibatch_size=2,
        sl_min_replay_buffer_size=0,
        sl_training_iterations=1,
    )

    config = SupervisedConfig(config_dict)
    agent_network = ModularAgentNetwork(
        config=config,
        num_actions=10,
        input_shape=(4,),
    )

    loss_pipeline = LossPipeline([ImitationLoss(config, device, num_actions=10)])
    loss_pipeline.validate_dependencies(
        network_output_keys={"policies"},
        target_keys={"target_policies"},
    )

    learner = UniversalLearner(
        config=config,
        agent_network=agent_network,
        device=device,
        num_actions=10,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=None,
        loss_pipeline=loss_pipeline,
        optimizer=MagicMock(),
    )
    return learner


def test_imitation_learner_loss_integration(imitation_setup):
    learner = imitation_setup

    batch = {
        "observations": torch.randn((2, 4)),
        "target_policies": torch.randint(0, 10, (2,)),
    }

    step_result = learner.compute_step_result(batch)

    assert isinstance(step_result.loss, torch.Tensor)
    assert "ImitationLoss" in step_result.loss_dict
