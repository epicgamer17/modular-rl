import pytest

pytestmark = pytest.mark.unit

import torch
from agents.action_selectors.selectors import (
    CategoricalSelector,
    EpsilonGreedySelector,
    ArgmaxSelector,
)
from modules.world_models.inference_output import InferenceOutput
from torch.distributions import Categorical


class MockNetwork:
    def obs_inference(self, obs):
        return None


def _setup_data():
    network = MockNetwork()
    batch_size = 4
    num_actions = 5
    q_values = torch.randn(batch_size, num_actions)
    logits = torch.randn(batch_size, num_actions)
    # Mock legal moves: Some None, some lists
    legal_moves = [[0, 1], None, [2, 3, 4], [1]]
    return network, batch_size, num_actions, q_values, logits, legal_moves


def test_mask_actions_2d():
    _, _, _, q_values, _, legal_moves = _setup_data()
    selector = ArgmaxSelector()
    masked = selector.mask_actions(q_values, legal_moves)

    # Row 0: only 0, 1 are legal
    assert not torch.isinf(masked[0, 0])
    assert not torch.isinf(masked[0, 1])
    assert torch.isinf(masked[0, 2])

    # Row 1: None means all illegal in current implementation
    assert torch.all(torch.isinf(masked[1]))


def test_epsilon_greedy_batched_exploration():
    network, _, num_actions, q_values, _, _ = _setup_data()
    # High epsilon to ensure some exploration
    selector = EpsilonGreedySelector(epsilon=1.0)
    output = InferenceOutput(q_values=q_values)

    # Test large batch to see independent exploration (stochastically)
    large_batch = 100
    large_q = torch.zeros(large_batch, num_actions)

    # We need info for legal moves
    info = {"legal_moves": [list(range(num_actions))] * large_batch}

    # Re-create output for large batch
    large_output = InferenceOutput(q_values=large_q)

    actions, _ = selector.select_action(
        network, None, info=info, network_output=large_output
    )

    # If it was not independent, all actions would be the same (either all greedy or all same random)
    # Since it uses torch.rand(batch_size), they should vary.
    assert len(torch.unique(actions)) > 1


def test_categorical_batched():
    network, batch_size, num_actions, _, _, _ = _setup_data()
    selector = CategoricalSelector(exploration=False)
    # High logits for specific actions
    custom_logits = torch.zeros(batch_size, num_actions)
    custom_logits[0, 0] = 10.0
    custom_logits[1, 1] = 10.0
    custom_logits[2, 2] = 10.0
    custom_logits[3, 3] = 10.0

    output = InferenceOutput(policy=Categorical(logits=custom_logits))
    actions, _ = selector.select_action(network, None, network_output=output)

    expected = torch.tensor([0, 1, 2, 3])
    assert torch.equal(actions, expected)
