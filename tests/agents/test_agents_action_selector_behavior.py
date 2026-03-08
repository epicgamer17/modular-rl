import pytest
pytestmark = pytest.mark.integration

import copy
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


class MockPolicyDist:
    def __init__(self, probs):
        self.probs = probs

    def sample(self):
        return torch.multinomial(self.probs, 1).squeeze(-1)

    def log_prob(self, action):
        return torch.log(self.probs.gather(-1, action.unsqueeze(-1)).squeeze(-1))


class MockInferenceOutput:
    def __init__(self, value=None, policy=None, logits=None):
        self.value = value
        self.policy = policy
        self.logits = logits
        self.network_state = None


class MockNetwork(nn.Module):
    def obs_inference(self, obs):
        return InferenceOutput(
            value=torch.tensor([0.0]),
            policy=None,
            reward=None,
            to_play=None,
            network_state=None,
            q_values=None,
        )


def _setup_action_selector_state():
    agent_network = MockNetwork()
    obs = torch.zeros(1, 4)
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    probs = torch.softmax(logits, dim=-1)
    value = torch.tensor([[0.5, 0.8, 0.1]])
    return agent_network, obs, logits, probs, value


def test_categorical_selector():
    agent_network, obs, _, probs, _ = _setup_action_selector_state()
    selector = CategoricalSelector(exploration=False)
    output = InferenceOutput(value=torch.tensor([0.0]), policy=MockPolicyDist(probs))
    action, _ = selector.select_action(agent_network, obs, network_output=output)
    assert action.item() == 2  # argmax of [1, 2, 3] is index 2


def test_argmax_selector():
    agent_network, obs, logits, _, value = _setup_action_selector_state()
    selector = ArgmaxSelector()
    # ArgmaxSelector looks for q_values
    output = InferenceOutput(
        value=value,
        policy=None,
        reward=None,
        to_play=None,
        network_state=None,
        q_values=logits,  # reusing logits as q_values for test
    )
    action, _ = selector.select_action(agent_network, obs, network_output=output)
    assert action.item() == 2  # argmax of logits [1.0, 2.0, 3.0] is index 2


def test_epsilon_greedy_selector():
    agent_network, obs, logits, _, value = _setup_action_selector_state()
    selector = EpsilonGreedySelector(epsilon=0.0)  # Greedy
    # Provide q_values explicitly
    output = InferenceOutput(
        value=value,
        policy=None,
        reward=None,
        to_play=None,
        network_state=None,
        q_values=logits,  # reusing logits as q_values
    )
    action, _ = selector.select_action(agent_network, obs, network_output=output)
    assert action.item() == 2  # argmax of [1.0, 2.0, 3.0] is index 2


def test_ppo_decorator():
    agent_network, obs, _, probs, value = _setup_action_selector_state()
    base_selector = CategoricalSelector(exploration=False)
    decorated_selector = PPODecorator(base_selector)
    output = InferenceOutput(policy=MockPolicyDist(probs), value=value)

    action, meta = decorated_selector.select_action(
        agent_network, obs, network_output=output
    )

    assert action.item() == 2
    assert "log_prob" in meta
    assert "value" in meta

    expected_log_prob = torch.log(probs[0, 2])
    assert torch.allclose(meta["log_prob"], expected_log_prob)


def test_factory_categorical_ppo(rainbow_cartpole_replay_config):
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
    assert not selector.inner_selector.default_exploration


def test_exploration_override():
    """Test that passing exploration arg overrides the default."""
    agent_network, obs, _, _, _ = _setup_action_selector_state()
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
            agent_network, obs, network_output=output, exploration=False
        )
        assert action.item() == 1
    except RuntimeError:
        pytest.fail("select_action called sample() despite exploration=False")

    # 2. Test exploration=True (Should call sample)
    try:
        selector.select_action(
            agent_network, obs, network_output=output, exploration=True
        )
    except RuntimeError:
        pass  # Expected
    else:
        pytest.fail("select_action should have called sample() with exploration=True")


def test_categorical_selector_masking():
    """Test that CategoricalSelector respects the action mask."""
    agent_network, obs, _, _, _ = _setup_action_selector_state()
    selector = CategoricalSelector(exploration=True)
    # Probabilities: [0.1, 0.2, 0.7] -> argmax is 2
    logits = torch.tensor([[1.0, 2.0, 3.4]])  # Logits for [0.1, 0.2, 0.7] approx
    probs = torch.softmax(logits, dim=-1)

    from torch.distributions import Categorical

    output = InferenceOutput(value=torch.tensor([0.0]), policy=Categorical(logits=logits))

    # Mask out index 2 (the most likely one)
    info = {"legal_moves": [[0, 1]]}

    # If it doesn't mask, it will likely pick 2 (70% of the time, but deterministic for argmax)
    # Since we use exploration=True, result is stochastic. Let's force exploration=False for deterministic test.
    action, _ = selector.select_action(
        agent_network,
        obs,
        info=info,
        network_output=output,
        exploration=False,
    )

    # Currently this will FAIL (it will return 2) if masking is not implemented.
    assert action.item() in [0, 1], f"Action {action.item()} should be in [0, 1]"


def test_categorical_selector_unmasked_log_prob():
    """Test that PPODecorator uses the masked distribution for log_probs."""
    agent_network, obs, _, _, _ = _setup_action_selector_state()
    from agents.action_selectors.decorators import PPODecorator
    from torch.distributions import Categorical

    base_selector = CategoricalSelector(exploration=False)
    selector = PPODecorator(base_selector)

    logits = torch.tensor([[1.0, 2.0, 3.0]])  # Probs are [.09, .24, .67]
    output = InferenceOutput(value=torch.tensor([0.0]), policy=Categorical(logits=logits))

    # Mask out 2. New probs should be [.09/(.09+.24), .24/(.09+.24), 0] -> approx [0.27, 0.73, 0]
    # Argmax should be 1.
    info = {"legal_moves": [[0, 1]]}

    action, meta = selector.select_action(
        agent_network,
        obs,
        info=info,
        network_output=output,
        exploration=False,
    )

    assert action.item() == 1
    # The log_prob should be against the MASKED distribution.
    # If unmasked: log(0.24) = -1.42
    # If masked: log(0.73) = -0.31

    # This test will also fail currently because log_prob will be calculated against original dist.
    unmasked_dist = Categorical(logits=logits)
    unmasked_log_prob = unmasked_dist.log_prob(action)

    # We want it to be DIFFERENT from unmasked log_prob (if masking worked)
    # But for now we just want to see it fail or record current state.
    assert meta["log_prob"].item() != unmasked_log_prob.item()


def test_nested_decorators_exploration_passing():
    """Test that decorators pass exploration arg down."""
    agent_network, obs, _, _, _ = _setup_action_selector_state()

    # Inner selector that fails if exploration is True (to verify it receives False)
    class MockInner(CategoricalSelector):
        def select_action(self, *args, exploration=True, **kwargs):
            if exploration:
                raise RuntimeError("Exploration was True")
            return torch.tensor(0), {}

    base = MockInner(exploration=True)
    decorator = PPODecorator(base)

    # Pass exploration=False, should NOT raise
    decorator.select_action(agent_network, obs, exploration=False)
