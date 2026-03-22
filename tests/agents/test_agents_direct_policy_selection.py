import pytest

pytestmark = pytest.mark.unit

import torch
from torch.distributions import Categorical

from agents.action_selectors.selectors import (
    ArgmaxSelector,
    CategoricalSelector,
    EpsilonGreedySelector,
)
from agents.action_selectors.types import InferenceResult
from modules.models.inference_output import InferenceOutput


def test_argmax_selector_picks_highest_q_value():
    selector = ArgmaxSelector()
    q_values = torch.tensor([[0.1, 0.5, 0.2, 0.2]])
    inf_result = InferenceResult(q_values=q_values)

    action, _ = selector.select_action(inf_result, {})

    assert action.item() == 1


def test_argmax_selector_respects_legal_moves():
    selector = ArgmaxSelector()
    q_values = torch.tensor([[0.1, 0.5, 0.2, 0.2]])
    inf_result = InferenceResult(q_values=q_values)

    # Mask out action 1 (highest q-value) — force to pick from [0, 2, 3]
    mask = torch.zeros(1, 4, dtype=torch.bool)
    mask[0, [0, 2, 3]] = True
    info = {"legal_moves_mask": mask}

    action, _ = selector.select_action(inf_result, info)

    assert action.item() == 2


def test_categorical_selector_greedy_mode():
    selector = CategoricalSelector(exploration=True)
    output = InferenceOutput(
        policy=Categorical(logits=torch.tensor([[0.0, 2.0, 0.0, 0.0]]))
    )
    inf_result = InferenceResult.from_inference_output(output)

    action, _ = selector.select_action(inf_result, {}, exploration=False)

    assert action.item() == 1


def test_epsilon_greedy_selector_can_disable_exploration():
    selector = EpsilonGreedySelector(epsilon=1.0)
    q_values = torch.tensor([[0.1, 0.5, 0.2, 0.2]])
    inf_result = InferenceResult(q_values=q_values)

    action, _ = selector.select_action(inf_result, {}, exploration=False)

    assert action.item() == 1
