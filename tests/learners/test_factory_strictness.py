import pytest
import torch
from agents.factories.learner import build_universal_learner, build_loss_pipeline
from modules.models.agent_network import AgentNetwork

pytestmark = pytest.mark.unit


class MockConfig:
    def __init__(self, agent_type=None):
        self.agent_type = agent_type
        self.minibatch_size = 2
        self.num_actions = 2
        self.input_shape = (4,)
        self.clip_param = 0.2  # PPO guessing hint
        self.unroll_steps = 3  # MuZero guessing hint
        self.clipnorm = 0

        class Game:
            def __init__(self):
                self.num_actions = 2

        self.game = Game()


def test_factory_fails_without_agent_type():
    config = MockConfig(agent_type=None)
    network = torch.nn.Module()
    network.num_actions = 2
    network.input_shape = (4,)
    device = torch.device("cpu")

    with pytest.raises(
        ValueError, match="config.agent_type must be explicitly defined."
    ):
        build_universal_learner(config, network, device)


def test_loss_pipeline_factory_fails_without_agent_type():
    config = MockConfig(agent_type=None)
    network = torch.nn.Module()
    device = torch.device("cpu")

    with pytest.raises(
        ValueError, match="config.agent_type must be explicitly defined."
    ):
        build_loss_pipeline(config, network, device)
