import pytest

pytestmark = pytest.mark.unit

import torch
from torch.distributions import Categorical

from agents.action_selectors.selectors import (
    ArgmaxSelector,
    CategoricalSelector,
    EpsilonGreedySelector,
)
from modules.world_models.inference_output import InferenceOutput


class MockNetwork:
    def obs_inference(self, obs):
        return InferenceOutput(q_values=torch.tensor([[0.1, 0.5, 0.2, 0.2]]))


def test_argmax_selector_picks_highest_q_value():
    selector = ArgmaxSelector()
    network = MockNetwork()
    obs = torch.zeros((1, 4))

    action, _ = selector.select_action(network, obs)

    assert action.item() == 1


def test_argmax_selector_respects_legal_moves():
    selector = ArgmaxSelector()
    network = MockNetwork()
    obs = torch.zeros((1, 4))

    action, _ = selector.select_action(network, obs, info={"legal_moves": [[0, 2, 3]]})

    assert action.item() == 2


def test_categorical_selector_greedy_mode():
    selector = CategoricalSelector(exploration=True)
    network = MockNetwork()
    obs = torch.zeros((1, 4))
    output = InferenceOutput(policy=Categorical(logits=torch.tensor([[0.0, 2.0, 0.0, 0.0]])))

    action, _ = selector.select_action(
        network,
        obs,
        network_output=output,
        exploration=False,
    )

    assert action.item() == 1


def test_epsilon_greedy_selector_can_disable_exploration():
    selector = EpsilonGreedySelector(epsilon=1.0)
    network = MockNetwork()
    obs = torch.zeros((1, 4))

    action, _ = selector.select_action(network, obs, exploration=False)

    assert action.item() == 1
