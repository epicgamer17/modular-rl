import torch
import numpy as np
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from agents.action_selectors.selectors import (
    EpsilonGreedy,
    TemperatureSelector,
    ArgmaxSelector,
    CategoricalSelector,
)
from agents.policies.policy import Policy
from agents.policies.search_policy import SearchPolicy
from agents.policies.direct_policy import DirectPolicy


def test_selectors():
    print("Testing EpsilonGreedy...")
    eg = EpsilonGreedy(epsilon=0.0)
    values = torch.tensor([[0.1, 0.5, 0.4]])
    assert eg.select(values) == 1, f"Expected 1, got {eg.select(values)}"
    eg.update_parameters({"epsilon": 0.5})
    assert eg.epsilon == 0.5
    print("EpsilonGreedy passed.")

    print("Testing TemperatureSelector...")
    ts = TemperatureSelector()
    probs = torch.tensor([[0.9, 0.1, 0.0]])
    # With low temp, it should pick index 0 overwhelmingly
    action = ts.select(probs, temperature=0.001)
    assert action == 0, f"Expected 0, got {action}"

    # Test temperature 0 (argmax)
    action_greedy = ts.select(probs, temperature=0)
    assert action_greedy == 0
    print("TemperatureSelector passed.")

    print("Testing ArgmaxSelector...")
    as_ = ArgmaxSelector()
    assert as_.select(torch.tensor([0.1, 0.5, 0.4])) == 1
    print("ArgmaxSelector passed.")


def test_policies():
    print("Testing Policy base class...")

    class MyPolicy(Policy):
        def reset(self, state):
            pass

        def compute_action(self, obs, info=None):
            return 0

    p = MyPolicy()
    p.update_parameters({})
    print("Policy base class passed.")


if __name__ == "__main__":
    try:
        test_selectors()
        test_policies()
        print("\nAll internal verifications passed!")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
