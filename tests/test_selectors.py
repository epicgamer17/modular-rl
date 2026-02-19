import unittest
import torch
import torch.nn as nn
from agents.action_selectors.decorators import PPODecorator
from agents.action_selectors.selectors import (
    CategoricalSelector,
    EpsilonGreedySelector,
    ArgmaxSelector,
)
from agents.action_selectors.factory import SelectorFactory
from configs.selectors import SelectorConfig
from modules.world_models.inference_output import InferenceOutput
from modules.world_models.inference_output import InferenceOutput


class MockPolicyDist:
    def __init__(self, probs):
        self.probs = probs

    def sample(self):
        return torch.multinomial(self.probs, 1).squeeze(-1)

    def log_prob(self, action):
        return torch.log(self.probs.gather(-1, action.unsqueeze(-1)).squeeze(-1))


class MockNetworkOutput:
    def __init__(self, value=None, policy=None, logits=None):
        self.value = value
        self.policy = policy
        self.logits = logits
        self.network_state = None


class MockNetwork(nn.Module):
    def initial_inference(self, obs):
        return InferenceOutput(value=torch.tensor([0.0]), policy=None)


class TestActionSelectors(unittest.TestCase):
    def setUp(self):
        self.agent_network = MockNetwork()
        self.obs = torch.zeros(1, 4)
        self.logits = torch.tensor([[1.0, 2.0, 3.0]])
        self.probs = torch.softmax(self.logits, dim=-1)
        self.value = torch.tensor([[0.5, 0.8, 0.1]])

    def test_categorical_selector(self):
        selector = CategoricalSelector(exploration=False)
        output = InferenceOutput(
            value=torch.tensor([0.0]), policy=MockPolicyDist(self.probs)
        )
        action, meta = selector.select_action(
            self.agent_network, self.obs, network_output=output
        )
        self.assertEqual(action.item(), 2)  # argmax of [1, 2, 3] is index 2

    def test_argmax_selector(self):
        selector = ArgmaxSelector()
        output = InferenceOutput(value=self.value, policy=None)
        action, meta = selector.select_action(
            self.agent_network, self.obs, network_output=output
        )
        self.assertEqual(action.item(), 1)  # argmax of [0.5, 0.8, 0.1] is index 1

    def test_epsilon_greedy_selector(self):
        # Test greedy choice (epsilon=0)
        selector = EpsilonGreedySelector(epsilon=0.0)
        output = InferenceOutput(value=self.value, policy=None)
        action, meta = selector.select_action(
            self.agent_network, self.obs, network_output=output
        )
        self.assertEqual(action.item(), 1)

    def test_ppo_decorator(self):
        base_selector = CategoricalSelector(exploration=False)
        decorated_selector = PPODecorator(base_selector)
        output = InferenceOutput(policy=MockPolicyDist(self.probs), value=self.value)

        action, meta = decorated_selector.select_action(
            self.agent_network, self.obs, network_output=output
        )

        self.assertEqual(action.item(), 2)
        self.assertIn("log_prob", meta)
        self.assertIn("value", meta)

        expected_log_prob = torch.log(self.probs[0, 2])
        self.assertTrue(torch.allclose(meta["log_prob"], expected_log_prob))

    def test_factory_categorical_ppo(self):
        config_dict = {
            "base": {"type": "categorical", "kwargs": {"exploration": False}},
            "decorators": [{"type": "ppo_injector", "kwargs": {}}],
        }

        selector = SelectorFactory.create(config_dict)
        self.assertIsInstance(selector, PPODecorator)
        self.assertIsInstance(selector.inner_selector, CategoricalSelector)
        self.assertFalse(selector.inner_selector.default_exploration)

    def test_exploration_override(self):
        """Test that passing exploration arg overrides the default."""
        # Default exploration=True
        selector = CategoricalSelector(exploration=True)

        # Mock policy
        class StrictPolicy:
            def __init__(self):
                self.probs = torch.tensor([[0.1, 0.9]])

            def sample(self):
                raise RuntimeError("Should not sample when exploration=False")

        output = InferenceOutput(value=torch.tensor([0.0]), policy=StrictPolicy())

        # 1. Test exploration=False (Should use argmax, not sample)
        try:
            action, _ = selector.select_action(
                self.agent_network, self.obs, network_output=output, exploration=False
            )
            self.assertEqual(action.item(), 1)
        except RuntimeError:
            self.fail("select_action called sample() despite exploration=False")

        # 2. Test exploration=True (Should call sample)
        try:
            selector.select_action(
                self.agent_network, self.obs, network_output=output, exploration=True
            )
        except RuntimeError:
            pass  # Expected
        else:
            self.fail("select_action should have called sample() with exploration=True")

    def test_nested_decorators_exploration_passing(self):
        """Test that decorators pass exploration arg down."""

        # Inner selector that fails if exploration is True (to verify it receives False)
        class MockInner(CategoricalSelector):
            def select_action(self, *args, exploration=True, **kwargs):
                if exploration:
                    raise RuntimeError("Exploration was True")
                return torch.tensor(0), {}

        base = MockInner(exploration=True)
        decorator = PPODecorator(base)

        # Pass exploration=False, should NOT raise
        decorator.select_action(self.agent_network, self.obs, exploration=False)


if __name__ == "__main__":
    unittest.main()
