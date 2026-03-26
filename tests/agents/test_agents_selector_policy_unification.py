import pytest

pytestmark = pytest.mark.integration

import copy
import torch
from torch.distributions import Categorical

from agents.action_selectors.decorators import PPODecorator
from agents.factories.action_selector import SelectorFactory
from agents.action_selectors.selectors import (
    ArgmaxSelector,
    CategoricalSelector,
    EpsilonGreedySelector,
)
from agents.action_selectors.types import InferenceResult
from modules.models.inference_output import InferenceOutput


def _make_result():
    logits = torch.tensor([[0.1, 2.0, 0.2]])
    q_values = logits.clone()
    return InferenceResult(
        value=torch.tensor([0.5]),
        q_values=q_values,
        logits=logits,
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
    inf_result = _make_result()

    action, meta = selector.select_action(inf_result, {})

    assert action.item() == 1
    assert "log_prob" in meta
    assert "value" in meta


def test_argmax_and_epsilon_greedy_agree_without_exploration():
    inf_result = _make_result()

    argmax_action, _ = ArgmaxSelector().select_action(inf_result, {})
    eg_action, _ = EpsilonGreedySelector(epsilon=1.0).select_action(
        inf_result,
        {},
        exploration=False,
    )

    assert argmax_action.item() == eg_action.item()
