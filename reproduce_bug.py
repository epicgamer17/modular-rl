import torch
import torch.nn as nn
from torch.distributions import Categorical
from agents.action_selectors.selectors import CategoricalSelector
from agents.action_selectors.decorators import PPODecorator
from collections import namedtuple

# Mock objects
NetworkOutput = namedtuple("NetworkOutput", ["policy", "value"])


class MockNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def obs_inference(self, obs):
        logits = torch.randn(1, 2)
        return NetworkOutput(
            policy=Categorical(logits=logits), value=torch.tensor([1.0])
        )


def test_categorical_selector_no_legal_moves():
    print("Testing CategoricalSelector without legal moves...")
    selector = CategoricalSelector()
    decorator = PPODecorator(selector)
    network = MockNetwork()
    obs = torch.randn(1, 4)
    info = {}  # No legal moves

    action, metadata = decorator.select_action(network, obs, info)

    assert "dist" in metadata, "dist should be in metadata"
    assert "log_prob" in metadata, "log_prob should be in metadata"
    print("Test passed!")


if __name__ == "__main__":
    test_categorical_selector_no_legal_moves()
