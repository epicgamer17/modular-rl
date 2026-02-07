import torch
import torch.nn as nn
import pytest
from agents.action_selectors.selectors import (
    ArgmaxSelector,
    CategoricalSelector,
    EpsilonGreedy as EpsilonGreedySelector,
)
from agents.policies.direct_policy import DirectPolicy


class MockModel(nn.Module):
    def __init__(self, output_size=4):
        super().__init__()
        self.output_size = output_size
        self.input_shape = (1, 10)
        self.fc = nn.Linear(10, output_size)

    def forward(self, x):
        # Return deterministic values for testing
        return torch.tensor([[0.1, 0.5, 0.2, 0.2]], dtype=torch.float32)


def test_argmax_selector():
    selector = ArgmaxSelector()
    predictions = torch.tensor([[0.1, 0.5, 0.2, 0.2]])

    # Test without info
    action = selector.select(predictions)
    assert action.item() == 1

    # Test with masking
    info = {"legal_moves": [0, 2, 3]}
    action = selector.select(predictions, info)
    assert (
        action.item() == 2
    )  # 0.2 is the max among [0.1, 0.2, 0.2] (argmax picks first)


def test_categorical_selector():
    selector = CategoricalSelector()
    probs = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    dist = torch.distributions.Categorical(probs=probs)

    action = selector.select(dist)
    assert action.item() == 1


def test_epsilon_greedy_selector():
    argmax_selector = ArgmaxSelector()

    # Epsilon = 0 (always greedy)
    selector = EpsilonGreedySelector(argmax_selector, epsilon=0.0)
    predictions = torch.tensor([[0.1, 0.5, 0.2, 0.2]])
    action = selector.select(predictions, exploration=False)
    assert action == 1

    # Epsilon = 1 (always random) - We can't easily test randomness deterministically,
    # but we can check if it calls the selector or random.
    # The epsilon_greedy_policy uses np.random.rand()


def test_direct_policy_integration():
    model = MockModel()
    selector = ArgmaxSelector()
    policy = DirectPolicy(model, selector)

    obs = torch.zeros((1, 10))
    action = policy.compute_action(obs)
    assert action.item() == 1


if __name__ == "__main__":
    pytest.main([__file__])
