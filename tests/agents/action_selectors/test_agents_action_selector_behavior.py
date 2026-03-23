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
from agents.action_selectors.types import InferenceResult
from configs.selectors import SelectorConfig
from modules.models.inference_output import InferenceOutput
from tests.agents.conftest import (
    MockPolicyDist,
    MockInferenceOutput,
    MockInferenceNetwork as MockNetwork,
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
    inf_result = InferenceResult.from_inference_output(output)
    action, _ = selector.select_action(inf_result, {})
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
    inf_result = InferenceResult.from_inference_output(output)
    action, _ = selector.select_action(inf_result, {})
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
    inf_result = InferenceResult.from_inference_output(output)
    action, _ = selector.select_action(inf_result, {})
    assert action.item() == 2  # argmax of [1.0, 2.0, 3.0] is index 2


def test_ppo_decorator():
    agent_network, obs, _, probs, value = _setup_action_selector_state()
    base_selector = CategoricalSelector(exploration=False)
    decorated_selector = PPODecorator(base_selector)
    output = InferenceOutput(policy=MockPolicyDist(probs), value=value)

    action, meta = decorated_selector.select_action(
        InferenceResult.from_inference_output(output), {}
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
    torch.manual_seed(42)
    # Default exploration=True
    selector = CategoricalSelector(exploration=True)

    probs = torch.tensor([[0.1, 0.9]])
    inf_result = InferenceResult(probs=probs, value=torch.tensor([0.0]))

    # 1. Test exploration=False (Should use argmax regardless of default)
    action, _ = selector.select_action(inf_result, {}, exploration=False)
    assert action.item() == 1, "exploration=False should pick argmax"

    # 2. Test exploration=True (Should sample; run many times to see non-determinism)
    actions = {
        selector.select_action(inf_result, {}, exploration=True)[0].item()
        for _ in range(50)
    }
    assert 0 in actions, "exploration=True should occasionally sample action 0"


def test_categorical_selector_masking():
    """Test that CategoricalSelector respects the action mask."""
    agent_network, obs, _, _, _ = _setup_action_selector_state()
    selector = CategoricalSelector(exploration=True)
    # Probabilities: [0.1, 0.2, 0.7] -> argmax is 2
    logits = torch.tensor([[1.0, 2.0, 3.4]])  # Logits for [0.1, 0.2, 0.7] approx
    probs = torch.softmax(logits, dim=-1)

    from torch.distributions import Categorical

    output = InferenceOutput(
        value=torch.tensor([0.0]), policy=Categorical(logits=logits)
    )

    # Mask out index 2 (the most likely one)
    mask = torch.tensor([[True, True, False]])
    info = {"legal_moves_mask": mask}

    # If it doesn't mask, it will likely pick 2 (70% of the time, but deterministic for argmax)
    # Since we use exploration=True, result is stochastic. Let's force exploration=False for deterministic test.
    inf_result = InferenceResult.from_inference_output(output)
    action, _ = selector.select_action(
        inf_result,
        info=info,
        exploration=False,
    )

    # Currently this will FAIL (it will return 2) if masking is not implemented.
    assert action.item() in [0, 1], f"Action {action.item()} should be in [0, 1]"


def test_categorical_selector_unmasked_log_prob():
    """Test that PPODecorator uses the masked distribution for log_prob."""
    agent_network, obs, _, _, _ = _setup_action_selector_state()
    from agents.action_selectors.decorators import PPODecorator
    from torch.distributions import Categorical

    base_selector = CategoricalSelector(exploration=False)
    selector = PPODecorator(base_selector)

    logits = torch.tensor([[1.0, 2.0, 3.0]])  # Probs are [.09, .24, .67]
    output = InferenceOutput(
        value=torch.tensor([0.0]), policy=Categorical(logits=logits)
    )

    # Mask out 2. New probs should be [.09/(.09+.24), .24/(.09+.24), 0] -> approx [0.27, 0.73, 0]
    # Argmax should be 1.
    mask = torch.tensor([[True, True, False]])
    info = {"legal_moves_mask": mask}

    inf_result = InferenceResult.from_inference_output(output)
    action, meta = selector.select_action(
        inf_result,
        info=info,
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
    inf_result = InferenceResult(
        logits=torch.tensor([[1.0]]), value=torch.tensor([0.0])
    )
    decorator.select_action(inf_result, {}, exploration=False)
