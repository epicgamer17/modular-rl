import pytest

pytestmark = pytest.mark.integration

import copy
import torch
from torch.distributions import Categorical

from agents.action_selectors.decorators import PPODecorator
from agents.action_selectors.factory import SelectorFactory
from agents.action_selectors.selectors import (
    ArgmaxSelector,
    CategoricalSelector,
    EpsilonGreedySelector,
)
from modules.world_models.inference_output import InferenceOutput


class MockNetwork:
    def obs_inference(self, obs):
        logits = torch.tensor([[0.1, 2.0, 0.2]])
        return InferenceOutput(
            value=torch.tensor([0.5]),
            q_values=logits,
            policy=Categorical(logits=logits),
        )


def test_selector_factory_builds_ppo_chain(rainbow_cartpole_replay_config):
    selector_config = copy.deepcopy(
        rainbow_cartpole_replay_config.config_dict["action_selector"]
    )
    selector_config["base"] = {
        "type": "categorical",
        "kwargs": {"exploration": False},
    }
    selector_config["decorators"] = [{"type": "ppo_injector", "kwargs": {}}]

    selector = SelectorFactory.create(selector_config)

    assert isinstance(selector, PPODecorator)
    assert isinstance(selector.inner_selector, CategoricalSelector)


def test_ppo_decorator_forwards_updates_to_inner_selector():
    inner = EpsilonGreedySelector(epsilon=0.1)
    selector = PPODecorator(inner)

    selector.update_parameters({"epsilon": 0.35})

    assert inner.epsilon == 0.35


def test_selector_stack_produces_action_and_metadata():
    selector = PPODecorator(CategoricalSelector(exploration=False))
    network = MockNetwork()
    obs = torch.zeros((1, 4))

    action, meta = selector.select_action(network, obs)

    assert action.item() == 1
    assert "log_prob" in meta
    assert "value" in meta


def test_argmax_and_epsilon_greedy_agree_without_exploration():
    network = MockNetwork()
    obs = torch.zeros((1, 4))

    argmax_action, _ = ArgmaxSelector().select_action(network, obs)
    eg_action, _ = EpsilonGreedySelector(epsilon=1.0).select_action(
        network,
        obs,
        exploration=False,
    )

    assert argmax_action.item() == eg_action.item()
