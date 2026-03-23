import pytest
import torch
import numpy as np
from types import SimpleNamespace
from torch.distributions import Categorical

from agents.action_selectors.selectors import (
    CategoricalSelector,
    EpsilonGreedySelector,
    ArgmaxSelector,
)
from agents.action_selectors.decorators import PPODecorator, TemperatureSelector
from agents.action_selectors.policy_sources import SearchPolicySource
from agents.action_selectors.types import InferenceResult
from modules.models.inference_output import InferenceOutput
from utils.schedule import ScheduleConfig

pytestmark = pytest.mark.unit


class MockNetwork(torch.nn.Module):
    def __init__(self, output_val=None):
        super().__init__()
        self.output_val = output_val
        self.input_shape = (4,)  # Dummy input shape

    def obs_inference(self, obs):
        return self.output_val


def test_categorical_selector_basic():
    """Test CategoricalSelector with temperature 0.0 (argmax) and 1.0 (sampling)."""
    torch.manual_seed(42)

    # [1, 3] logits -> probs approx [0.12, 0.88]
    logits = torch.tensor([[1.0, 3.0]])
    policy = Categorical(logits=logits)
    output = InferenceOutput(policy=policy, value=torch.tensor([0.0]))
    network = MockNetwork()

    # 1. Greedy (exploration=False)
    selector_greedy = CategoricalSelector(exploration=False)
    inf_result = InferenceResult.from_inference_output(output)
    action, _ = selector_greedy.select_action(inf_result, {})
    assert action.item() == 1, "Should pick argmax action"

    # 2. Sampling (exploration=True)
    selector_sample = CategoricalSelector(exploration=True)
    # With seed 42, check we get some distribution
    actions = []
    for _ in range(100):
        action, _ = selector_sample.select_action(inf_result, {})
        actions.append(action.item())

    assert (
        0 in actions and 1 in actions
    ), "Should sample both actions over multiple tries"
    assert sum(actions) > 50, "Should favor action 1 (higher probability)"


def test_categorical_selector_with_inference_result():
    """Test CategoricalSelector using InferenceResult directly (logits field)."""
    logits = torch.tensor([[1.0, 3.0]])
    inf_result = InferenceResult(logits=logits, value=torch.tensor([0.0]))

    selector = CategoricalSelector(exploration=False)
    action, _ = selector.select_action(inf_result, {})
    assert action.item() == 1, "Should pick argmax action from InferenceResult"


def test_categorical_selector_with_probs():
    """Test CategoricalSelector falls back to probs when logits is None."""
    probs = torch.tensor([[0.1, 0.9]])
    inf_result = InferenceResult(probs=probs, value=torch.tensor([0.0]))

    selector = CategoricalSelector(exploration=False)
    action, _ = selector.select_action(inf_result, {})
    assert action.item() == 1, "Should pick argmax action from probs"


def test_categorical_selector_masking():
    """Test CategoricalSelector respects action masks."""
    logits = torch.tensor([[10.0, 1.0]])  # 0 is very likely
    policy = Categorical(logits=logits)
    output = InferenceOutput(policy=policy, value=torch.tensor([0.0]))

    # Mask out index 0
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[0, 1] = True
    info = {"legal_moves_mask": mask}

    selector = CategoricalSelector(exploration=False)
    inf_result = InferenceResult.from_inference_output(output)

    action, meta = selector.select_action(inf_result, info)
    assert action.item() == 1, "Should pick action 1 because 0 is masked"
    assert "policy_dist" in meta
    assert (
        torch.isinf(meta["policy_dist"].logits[0, 0])
        and meta["policy_dist"].logits[0, 0] < 0
    )


def test_epsilon_greedy_selector_basic():
    """Test EpsilonGreedySelector with epsilon 0.0 (greedy) and 1.0 (random)."""
    torch.manual_seed(42)
    np.random.seed(42)

    q_values = torch.tensor([[1.0, 5.0, 2.0]])
    output = InferenceOutput(q_values=q_values, value=torch.tensor([0.0]))

    # 1. Pure Greedy (epsilon=0.0)
    selector_greedy = EpsilonGreedySelector(epsilon=0.0)
    inf_result = InferenceResult.from_inference_output(output)
    action, _ = selector_greedy.select_action(inf_result, {})
    assert action.item() == 1, "Should pick argmax of q_values"

    # 2. Pure Random (epsilon=1.0)
    selector_random = EpsilonGreedySelector(epsilon=1.0)
    actions = set()
    for _ in range(100):
        action, _ = selector_random.select_action(inf_result, {})
        actions.add(action.item())
    assert actions == {0, 1, 2}, "Should sample all actions with high epsilon"


def test_epsilon_greedy_masking():
    """Test EpsilonGreedySelector respects masking during both greedy and random phases."""
    torch.manual_seed(42)
    np.random.seed(42)

    q_values = torch.tensor([[10.0, 1.0]])  # 0 is greedy choice
    output = InferenceOutput(q_values=q_values, value=torch.tensor([0.0]))

    # Mask out 0, force 1
    mask = torch.zeros_like(q_values, dtype=torch.bool)
    mask[0, 1] = True
    info = {"legal_moves_mask": mask}
    inf_result = InferenceResult.from_inference_output(output)

    # Test Greedy Masking (epsilon=0)
    selector_greedy = EpsilonGreedySelector(epsilon=0.0)
    action, _ = selector_greedy.select_action(inf_result, info)
    assert action.item() == 1

    # Random masking (epsilon=1.0)
    selector_random = EpsilonGreedySelector(epsilon=1.0)
    for _ in range(20):
        action, _ = selector_random.select_action(inf_result, info)
        assert action.item() == 1, "Random selection should also respect mask"


def test_argmax_selector():
    """Test ArgmaxSelector across different input types (q_values, probs, logits)."""
    # 1. Q-Values
    out_q = InferenceOutput(
        q_values=torch.tensor([[1.0, 5.0]]), value=torch.tensor([0.0])
    )
    selector = ArgmaxSelector()
    inf_q = InferenceResult.from_inference_output(out_q)
    action, _ = selector.select_action(inf_q, {})
    assert action.item() == 1

    # 2. Policy Probs
    out_p = InferenceOutput(
        policy=Categorical(probs=torch.tensor([[0.9, 0.1]])), value=torch.tensor([0.0])
    )
    inf_p = InferenceResult.from_inference_output(out_p)
    action, _ = selector.select_action(inf_p, {})
    assert action.item() == 0

    # 3. Masking
    mask = torch.tensor([[False, True]])
    info = {"legal_moves_mask": mask}
    action, _ = selector.select_action(inf_q, info)
    assert action.item() == 1


def test_ppo_decorator():
    """Test PPODecorator injects log_prob and value."""
    logits = torch.tensor([[1.0, 2.0]])
    policy = Categorical(logits=logits)
    value = torch.tensor([[0.5]])
    output = InferenceOutput(policy=policy, value=value)

    inner = CategoricalSelector(exploration=False)
    selector = PPODecorator(inner)

    inf_result = InferenceResult.from_inference_output(output)
    action, meta = selector.select_action(inf_result, {})

    assert action.item() == 1
    assert "log_prob" in meta
    assert "value" in meta
    assert torch.allclose(meta["value"], value.cpu())

    expected_log_prob = policy.log_prob(action)
    assert torch.allclose(meta["log_prob"], expected_log_prob.cpu())


class MockSearch:
    def __init__(self, config):
        self.config = config

    def run(self, obs, info, agent_network, trajectory_action=None, exploration=True):
        # returns root_value, exploratory_policy, target_policy, best_action, search_metadata
        return (
            0.5,
            torch.tensor([0.1, 0.9]),
            torch.tensor([0.0, 1.0]),
            1,
            {"mcts_simulations": 10},
        )

    def run_vectorized(self, obs, infos, agent_network, trajectory_actions=None):
        # returns root_values, exploratory_policies, target_policies, best_actions, sm_list
        B = obs.shape[0]
        return (
            [0.5] * B,
            [torch.tensor([0.1, 0.9])] * B,
            [torch.tensor([0.0, 1.0])] * B,
            [1] * B,
            [{}] * B,
        )


def test_search_policy_source():
    """Test SearchPolicySource returns InferenceResult with probs from MCTS."""
    torch.manual_seed(42)
    network = MockNetwork()
    config = SimpleNamespace(
        num_simulations=10,
        temperature_schedule=ScheduleConfig(type="constant", initial=1.0),
    )
    search = MockSearch(config)
    source = SearchPolicySource(search_engine=search, agent_network=network, config=config)

    obs = torch.randn(1, 4)
    result = source.get_inference(obs, {}, to_play=0)

    assert result.probs is not None, "SearchPolicySource must populate probs"
    assert result.logits is None, "SearchPolicySource should not populate logits"
    assert result.value is not None
    assert result.probs.shape == torch.Size([1, 2])
    assert torch.allclose(result.probs, torch.tensor([0.1, 0.9]))


def test_search_policy_source_agent_network_kwarg():
    """Test SearchPolicySource accepts agent_network via kwarg."""
    network = MockNetwork()
    config = SimpleNamespace(
        num_simulations=10,
        temperature_schedule=ScheduleConfig(type="constant", initial=1.0),
    )
    search = MockSearch(config)
    # Initialize with None; expect kwarg to be used instead
    source = SearchPolicySource(search_engine=search, agent_network=None, config=config)

    obs = torch.randn(1, 4)
    result = source.get_inference(obs, {}, agent_network=network, to_play=0)

    assert result.probs is not None
    assert torch.allclose(result.probs, torch.tensor([0.1, 0.9]))


def test_temperature_selector_identity():
    """Test TemperatureSelector with temp=1.0 is identity (no change to logits)."""
    torch.manual_seed(42)
    probs = torch.tensor([[0.1, 0.9]])
    inf_result = InferenceResult(probs=probs, value=torch.tensor([0.0]))

    config = SimpleNamespace(
        temperature_schedule=ScheduleConfig(type="constant", initial=1.0)
    )
    inner = CategoricalSelector(exploration=False)
    selector = TemperatureSelector(inner, config)

    action, _ = selector.select_action(inf_result, {}, episode_step=0)
    # With temp=1.0, logits = log(probs), argmax should still be 1
    assert action.item() == 1


def test_temperature_selector_greedy():
    """Test TemperatureSelector with temp=0.0 always picks best action."""
    torch.manual_seed(42)
    probs = torch.tensor([[0.1, 0.9]])
    inf_result = InferenceResult(probs=probs, value=torch.tensor([0.0]))

    config = SimpleNamespace(
        temperature_schedule=ScheduleConfig(type="constant", initial=0.0)
    )
    inner = CategoricalSelector(exploration=True)  # would sample, but temp overrides
    selector = TemperatureSelector(inner, config)

    for _ in range(10):
        action, _ = selector.select_action(inf_result, {}, episode_step=0)
        assert action.item() == 1, "With temp=0, always pick best action"


def test_temperature_selector_exploration_false():
    """Test TemperatureSelector forces temp=0.0 when exploration=False."""
    torch.manual_seed(42)
    probs = torch.tensor([[0.1, 0.9]])
    inf_result = InferenceResult(probs=probs, value=torch.tensor([0.0]))

    config = SimpleNamespace(
        temperature_schedule=ScheduleConfig(type="constant", initial=1.0)
    )
    inner = CategoricalSelector(exploration=True)
    selector = TemperatureSelector(inner, config)

    for _ in range(10):
        action, _ = selector.select_action(
            inf_result, {}, exploration=False, episode_step=0
        )
        assert (
            action.item() == 1
        ), "exploration=False forces argmax regardless of schedule"


def test_temperature_selector_from_logits():
    """Test TemperatureSelector works correctly starting from raw logits."""
    logits = torch.tensor([[1.0, 3.0]])
    inf_result = InferenceResult(logits=logits, value=torch.tensor([0.0]))

    config = SimpleNamespace(
        temperature_schedule=ScheduleConfig(type="constant", initial=1.0)
    )
    inner = CategoricalSelector(exploration=False)
    selector = TemperatureSelector(inner, config)

    action, _ = selector.select_action(inf_result, {}, episode_step=0)
    assert action.item() == 1


def test_batched_epsilon_greedy():
    """Test EpsilonGreedySelector with batch inputs."""
    q_values = torch.tensor([[10.0, 1.0], [1.0, 10.0]])
    output = InferenceOutput(q_values=q_values, value=torch.tensor([0.0, 0.0]))

    selector = EpsilonGreedySelector(epsilon=0.0)
    inf_result = InferenceResult.from_inference_output(output)
    actions, _ = selector.select_action(inf_result, {})

    assert actions[0].item() == 0
    assert actions[1].item() == 1
    assert actions.shape == (2,)
