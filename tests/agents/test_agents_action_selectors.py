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
from agents.action_selectors.decorators import PPODecorator, MCTSDecorator
from modules.world_models.inference_output import InferenceOutput
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
    action, _ = selector_greedy.select_action(network, None, network_output=output)
    assert action.item() == 1, "Should pick argmax action"

    # 2. Sampling (exploration=True)
    selector_sample = CategoricalSelector(exploration=True)
    # With seed 42, check we get some distribution
    actions = []
    for _ in range(100):
        action, _ = selector_sample.select_action(network, None, network_output=output)
        actions.append(action.item())

    assert (
        0 in actions and 1 in actions
    ), "Should sample both actions over multiple tries"
    assert sum(actions) > 50, "Should favor action 1 (higher probability)"


def test_categorical_selector_masking():
    """Test CategoricalSelector respects action masks."""
    logits = torch.tensor([[10.0, 1.0]])  # 0 is very likely
    policy = Categorical(logits=logits)
    output = InferenceOutput(policy=policy, value=torch.tensor([0.0]))
    network = MockNetwork()

    # Mask out index 0
    info = {"legal_moves": [[1]]}
    selector = CategoricalSelector(exploration=False)

    action, meta = selector.select_action(
        network, None, info=info, network_output=output
    )
    assert action.item() == 1, "Should pick action 1 because 0 is masked"
    assert "policy" in meta
    assert torch.isinf(meta["policy"].logits[0, 0]) and meta["policy"].logits[0, 0] < 0


def test_epsilon_greedy_selector_basic():
    """Test EpsilonGreedySelector with epsilon 0.0 (greedy) and 1.0 (random)."""
    torch.manual_seed(42)
    np.random.seed(42)

    q_values = torch.tensor([[1.0, 5.0, 2.0]])
    output = InferenceOutput(q_values=q_values, value=torch.tensor([0.0]))
    network = MockNetwork()

    # 1. Pure Greedy (epsilon=0.0)
    selector_greedy = EpsilonGreedySelector(epsilon=0.0)
    action, _ = selector_greedy.select_action(network, None, network_output=output)
    assert action.item() == 1, "Should pick argmax of q_values"

    # 2. Pure Random (epsilon=1.0)
    selector_random = EpsilonGreedySelector(epsilon=1.0)
    actions = set()
    for _ in range(100):
        action, _ = selector_random.select_action(network, None, network_output=output)
        actions.add(action.item())
    assert actions == {0, 1, 2}, "Should sample all actions with high epsilon"


def test_epsilon_greedy_masking():
    """Test EpsilonGreedySelector respects masking during both greedy and random phases."""
    torch.manual_seed(42)
    np.random.seed(42)

    q_values = torch.tensor([[10.0, 1.0]])  # 0 is greedy choice
    output = InferenceOutput(q_values=q_values, value=torch.tensor([0.0]))
    network = MockNetwork()

    # Mask out 0, force 1
    info = {"legal_moves": [[1]]}

    # Test Greedy Masking (epsilon=0)
    selector_greedy = EpsilonGreedySelector(epsilon=0.0)
    action, _ = selector_greedy.select_action(
        network, None, info=info, network_output=output
    )
    assert action.item() == 1

    # Random masking (epsilon=1.0)
    selector_random = EpsilonGreedySelector(epsilon=1.0)
    for _ in range(20):
        action, _ = selector_random.select_action(
            network, None, info=info, network_output=output
        )
        assert action.item() == 1, "Random selection should also respect mask"


def test_argmax_selector():
    """Test ArgmaxSelector across different input types (q_values, probs, logits)."""
    network = MockNetwork()

    # 1. Q-Values
    out_q = InferenceOutput(
        q_values=torch.tensor([[1.0, 5.0]]), value=torch.tensor([0.0])
    )
    selector = ArgmaxSelector()
    action, _ = selector.select_action(network, None, network_output=out_q)
    assert action.item() == 1

    # 2. Policy Probs
    out_p = InferenceOutput(
        policy=Categorical(probs=torch.tensor([[0.9, 0.1]])), value=torch.tensor([0.0])
    )
    action, _ = selector.select_action(network, None, network_output=out_p)
    assert action.item() == 0

    # 3. Masking
    info = {"legal_moves": [[1]]}
    action, _ = selector.select_action(network, None, info=info, network_output=out_q)
    assert action.item() == 1


def test_ppo_decorator():
    """Test PPODecorator injects log_prob and value."""
    logits = torch.tensor([[1.0, 2.0]])
    policy = Categorical(logits=logits)
    value = torch.tensor([[0.5]])
    output = InferenceOutput(policy=policy, value=value)
    network = MockNetwork()

    inner = CategoricalSelector(exploration=False)
    selector = PPODecorator(inner)

    action, meta = selector.select_action(network, None, network_output=output)

    assert action.item() == 1
    assert "log_prob" in meta
    assert "value" in meta
    assert torch.allclose(meta["value"], value.cpu())

    expected_log_prob = policy.log_prob(action)
    assert torch.allclose(meta["log_prob"], expected_log_prob.cpu())


class MockSearch:
    def __init__(self, config):
        self.config = config

    def run(self, obs, info, to_play, network, exploration=True):
        # returns root_value, exploratory_policy, target_policy, best_action, search_metadata
        return (
            0.5,
            torch.tensor([0.1, 0.9]),
            torch.tensor([0.0, 1.0]),
            1,
            {"mcts_simulations": 10},
        )

    def run_vectorized(self, obs, infos, to_play, network):
        # returns root_values, exploratory_policies, target_policies, best_actions, sm_list
        B = obs.shape[0]
        return (
            [0.5] * B,
            [torch.tensor([0.1, 0.9])] * B,
            [torch.tensor([0.0, 1.0])] * B,
            [1] * B,
            [{}] * B,
        )


def test_mcts_decorator():
    """Test MCTSDecorator temperature handling and search delegation."""
    network = MockNetwork()
    config = SimpleNamespace(
        num_simulations=10,
        temperature_schedule=ScheduleConfig(type="constant", initial=1.0),
    )
    search = MockSearch(config)
    inner = CategoricalSelector(exploration=False)

    selector = MCTSDecorator(inner, search, config)

    # Test temperature 0.0 (greedy)
    obs = torch.randn(1, 4)
    action, meta = selector.select_action(network, obs, exploration=False)

    assert action.item() == 1
    assert meta["value"] == 0.5
    assert meta["best_action"] == 1

    # Test batching
    obs_batch = torch.randn(2, 4)
    info_batch = {"legal_moves": [[0, 1], [0, 1]], "player": [0, 0]}
    action_batch, meta_batch = selector.select_action(
        network, obs_batch, info=info_batch, player_id=[0, 0]
    )
    assert action_batch.shape == (2,)
    assert len(meta_batch["value"]) == 2


def test_batched_epsilon_greedy():
    """Test EpsilonGreedySelector with batch inputs."""
    q_values = torch.tensor([[10.0, 1.0], [1.0, 10.0]])
    output = InferenceOutput(q_values=q_values, value=torch.tensor([0.0, 0.0]))
    network = MockNetwork()

    selector = EpsilonGreedySelector(epsilon=0.0)
    actions, _ = selector.select_action(network, None, network_output=output)

    assert actions[0].item() == 0
    assert actions[1].item() == 1
    assert actions.shape == (2,)
