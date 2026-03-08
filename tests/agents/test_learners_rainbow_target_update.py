import pytest
import torch
import numpy as np
from agents.learners.rainbow_learner import RainbowLearner
from configs.agents.rainbow_dqn import RainbowConfig

pytestmark = pytest.mark.unit


class SimpleNet(torch.nn.Module):
    """A lightweight dummy network to test parameter synchronization."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def learner_inference(self, batch):
        pass

    def reset_noise(self):
        pass


def test_rainbow_learner_hard_update(make_rainbow_config_dict, cartpole_game_config):
    torch.manual_seed(42)
    np.random.seed(42)

    # Force soft_update to False to test the hard update branch
    config_dict = make_rainbow_config_dict(soft_update=False)
    config = RainbowConfig(config_dict, cartpole_game_config)

    agent_net = SimpleNet()
    target_net = SimpleNet()

    # Artificially desynchronize the weights
    with torch.no_grad():
        agent_net.linear.weight.fill_(1.0)
        target_net.linear.weight.fill_(0.0)

    learner = RainbowLearner(
        config=config,
        agent_network=agent_net,
        target_agent_network=target_net,
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
    )

    # Perform the hard update
    learner.update_target_network()

    # Verify target net explicitly matches online net
    assert torch.allclose(target_net.linear.weight, agent_net.linear.weight)
    assert torch.allclose(target_net.linear.weight, torch.tensor(1.0))
