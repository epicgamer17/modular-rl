import torch
import torch.nn as nn
from modules.action_selectors import (
    ArgmaxSelector,
    CategoricalSelector,
    EpsilonGreedySelector,
)
from modules.direct_policy import DirectPolicy


class MockModel(nn.Module):
    def __init__(self, output_size=4):
        super().__init__()
        self.output_size = output_size
        self.input_shape = (1, 10)
        self.fc = nn.Linear(10, output_size)

    def forward(self, x):
        return torch.tensor([[0.1, 0.5, 0.2, 0.2]], dtype=torch.float32)


def test_argmax_selector():
    print("Testing ArgmaxSelector...")
    selector = ArgmaxSelector()
    predictions = torch.tensor([[0.1, 0.5, 0.2, 0.2]])

    action = selector.select_action(predictions)
    assert action.item() == 1

    info = {"legal_moves": [[0, 2, 3]]}  # action_mask expects list of lists for batch
    action = selector.select_action(predictions, info)
    assert action.item() == 2
    print("ArgmaxSelector passed!")


def test_categorical_selector():
    print("Testing CategoricalSelector...")
    selector = CategoricalSelector()
    probs = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    dist = torch.distributions.Categorical(probs=probs)

    action = selector.select_action(dist)
    assert action.item() == 1
    print("CategoricalSelector passed!")


def test_epsilon_greedy_selector():
    print("Testing EpsilonGreedySelector...")
    argmax_selector = ArgmaxSelector()
    selector = EpsilonGreedySelector(argmax_selector, epsilon=0.0)
    predictions = torch.tensor([[0.1, 0.5, 0.2, 0.2]])
    action = selector.select_action(predictions, {})
    assert action == 1
    print("EpsilonGreedySelector passed!")


def test_direct_policy_integration():
    print("Testing DirectPolicy integration...")
    model = MockModel()
    selector = ArgmaxSelector()
    policy = DirectPolicy(model, selector)

    obs = torch.zeros((1, 10))
    action = policy.compute_action(obs)
    assert action.item() == 1
    print("DirectPolicy integration passed!")


if __name__ == "__main__":
    test_argmax_selector()
    test_categorical_selector()
    test_epsilon_greedy_selector()
    test_direct_policy_integration()
    print("All tests passed!")
