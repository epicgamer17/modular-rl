import unittest
import torch
import numpy as np
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


class TestBatchedActionSelectors(unittest.TestCase):
    def setUp(self):
        self.network = MockNetwork()
        self.batch_size = 4
        self.num_actions = 5
        self.q_values = torch.randn(self.batch_size, self.num_actions)
        self.logits = torch.randn(self.batch_size, self.num_actions)
        # Mock legal moves: Some None, some lists
        self.legal_moves = [[0, 1], None, [2, 3, 4], [1]]

    def test_mask_actions_2d(self):
        selector = ArgmaxSelector()
        masked = selector.mask_actions(self.q_values, self.legal_moves)

        # Row 0: only 0, 1 are legal
        self.assertFalse(torch.isinf(masked[0, 0]))
        self.assertFalse(torch.isinf(masked[0, 1]))
        self.assertTrue(torch.isinf(masked[0, 2]))

        # Row 1: None means all illegal in current implementation
        self.assertTrue(torch.all(torch.isinf(masked[1])))

    def test_epsilon_greedy_batched_exploration(self):
        # High epsilon to ensure some exploration
        selector = EpsilonGreedySelector(epsilon=1.0)
        output = InferenceOutput(q_values=self.q_values)

        # Test large batch to see independent exploration (stochastically)
        large_batch = 100
        large_q = torch.zeros(large_batch, self.num_actions)

        # We need info for legal moves
        info = {"legal_moves": [list(range(self.num_actions))] * large_batch}

        # Re-create output for large batch
        large_output = InferenceOutput(q_values=large_q)

        actions, _ = selector.select_action(
            self.network, None, info=info, network_output=large_output
        )

        # If it was not independent, all actions would be the same (either all greedy or all same random)
        # Since it uses torch.rand(batch_size), they should vary.
        self.assertTrue(len(torch.unique(actions)) > 1)

    def test_categorical_batched(self):
        selector = CategoricalSelector(exploration=False)
        # High logits for specific actions
        custom_logits = torch.zeros(self.batch_size, self.num_actions)
        custom_logits[0, 0] = 10.0
        custom_logits[1, 1] = 10.0
        custom_logits[2, 2] = 10.0
        custom_logits[3, 3] = 10.0

        output = InferenceOutput(policy=Categorical(logits=custom_logits))
        actions, _ = selector.select_action(self.network, None, network_output=output)

        expected = torch.tensor([0, 1, 2, 3])
        self.assertTrue(torch.equal(actions, expected))


if __name__ == "__main__":
    unittest.main()
