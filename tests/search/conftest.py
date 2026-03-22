"""Shared test doubles for search tests.

Provides unified mock/dummy classes used across search unit tests,
eliminating duplication of DummyMinMaxStats, DummyChild, DummyNode, etc.
"""

import pytest
import torch
from typing import NamedTuple, Optional, Dict, Any
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Search Tree Dummies (used by scoring, selector, pruner, backprop, root policy tests)
# ---------------------------------------------------------------------------


class DummyMinMaxStats:
    """Configurable MinMaxStats stub.

    Variants:
    - DummyMinMaxStats()                → no-op (for selectors/backprop)
    - DummyMinMaxStats(normalize=True)  → identity normalize (return x)
    - DummyMinMaxStats(normalize_fn=fn) → custom normalize function
    """

    def __init__(self, normalize: bool = False, normalize_fn=None):
        self._normalize = normalize
        self._normalize_fn = normalize_fn

    def normalize(self, val):
        if self._normalize_fn is not None:
            return self._normalize_fn(val)
        return val

    def update(self, val):
        pass


class DummyChild:
    """Configurable child node stub for search tree tests.

    Supports all variants used across test files:
    - scoring tests: expanded, val, prior, visits
    - pruner tests: is_expanded, visits
    - root policy tests: expanded, val
    - selector sampling tests: idx
    """

    def __init__(self, expanded=True, val=0.0, prior=0.1, visits=1, idx=None):
        self._expanded = expanded
        self._val = val
        self.prior = prior
        self.visits = visits
        self.idx = idx

    def expanded(self):
        return self._expanded

    def value(self):
        return self._val


class DummyScoringMethod:
    """Stub scoring method that returns pre-configured scores."""

    def __init__(self, scores: torch.Tensor):
        self._scores = scores

    def get_scores(self, node, min_max_stats):
        return self._scores.clone()


class DummySearchConfig:
    """Lightweight search config stub with Gumbel parameters."""

    def __init__(self, gumbel_cvisit: float = 50.0, gumbel_cscale: float = 1.0):
        self.gumbel_cvisit = gumbel_cvisit
        self.gumbel_cscale = gumbel_cscale


class DummyBackpropConfig:
    """Config stub for backpropagation tests with game and discount settings."""

    def __init__(self, num_players: int = 2, discount_factor: float = 0.9):
        self.game = SimpleNamespace(num_players=num_players)
        self.discount_factor = discount_factor


# ---------------------------------------------------------------------------
# Search Node Factories
# ---------------------------------------------------------------------------


def make_dummy_scoring_node(
    visits: int = 10,
    child_visits=None,
    child_priors=None,
    network_policy=None,
    child_values=None,
    children: Optional[Dict[int, DummyChild]] = None,
    node_value: float = 0.5,
    v_mix: float = 0.25,
    unvisited_q: float = -1.0,
    child_q_offset: float = 1.0,
):
    """Factory for DummyNode used in scoring/root-policy tests.

    Returns a SimpleNamespace with the interface expected by scoring methods.
    """

    node = SimpleNamespace(
        visits=visits,
        pb_c_base=19652,
        pb_c_init=1.25,
        child_visits=child_visits if child_visits is not None else torch.tensor([5.0, 0.0]),
        child_priors=child_priors if child_priors is not None else torch.tensor([0.7, 0.3]),
        network_policy=network_policy if network_policy is not None else torch.tensor([0.7, 0.3]),
        child_values=child_values if child_values is not None else torch.tensor([1.5, 0.0]),
        children=children
        if children is not None
        else {
            0: DummyChild(val=1.5),
            1: DummyChild(expanded=False, val=0.0, visits=0),
        },
    )
    node.value = lambda: node_value
    node.get_v_mix = lambda: v_mix
    node.get_child_q_from_parent = lambda child: child.value() + child_q_offset
    node.get_child_q_for_unvisited = lambda: unvisited_q
    return node


def make_dummy_pruner_node(child_q_values: Dict[int, float]):
    """Factory for DummyNode used in pruner tests.

    Each action maps to a Q-value; children are auto-created.
    """

    node = SimpleNamespace(
        children={action: DummyChild() for action in child_q_values},
        child_q_values=child_q_values,
    )

    def _get_child_q(child):
        for act, ch in node.children.items():
            if ch is child:
                return child_q_values[act]
        return 0.0

    node.get_child_q_from_parent = _get_child_q
    return node


def make_dummy_utils_node():
    """Factory for DummyNode used in search utility tests (get_completed_q, etc.)."""

    node = SimpleNamespace(
        child_priors=torch.tensor([0.1, 0.7, 0.2]),
        network_policy=torch.tensor([0.2, 0.6, 0.2]),
        child_visits=torch.tensor([0, 5, 2]),
        child_values=torch.tensor([0.0, 5.0, 2.0]),
    )
    node.get_v_mix = lambda: torch.tensor(1.0)
    node.get_child_q_for_unvisited = lambda: torch.tensor(-1.0)
    return node


def make_dummy_root_node():
    """Factory for DummyRootNode used in root policy tests."""

    node = SimpleNamespace(
        child_visits=torch.tensor([10.0, 20.0, 0.0]),
        child_values=torch.tensor([0.5, 0.8, 0.0]),
        child_priors=torch.tensor([0.3, 0.6, 0.1]),
        network_policy=torch.tensor([0.3, 0.6, 0.1]),
        children={
            0: DummyChild(True, 0.5),
            1: DummyChild(True, 0.8),
            2: DummyChild(False, 0.0),
        },
    )
    node.get_child_q_from_parent = lambda child: child.value()
    node.get_child_q_for_unvisited = lambda: 0.1
    node.get_v_mix = lambda: 0.2
    return node


# ---------------------------------------------------------------------------
# Mock Network for MCTS (deterministic, descending logits)
# ---------------------------------------------------------------------------


class MockNetworkState(NamedTuple):
    """Opaque token compatible with both Python and AOS MCTS backends."""

    data: torch.Tensor  # shape [B, D]; corresponds to the latent state
    wm_memory: Optional[torch.Tensor] = None

    @classmethod
    def batch(cls, states: list) -> "MockNetworkState":
        return cls(data=torch.stack([s.data for s in states], dim=0))

    def unbatch(self) -> list:
        return [MockNetworkState(data=self.data[i]) for i in range(self.data.shape[0])]


class MockSearchNetwork(torch.nn.Module):
    """Stateless deterministic network for search parity/integration tests.

    Returns descending logits [A, A-1, ..., 1] so action 0 is always preferred.
    """

    def __init__(self, num_actions: int, mock_value: float = 0.5):
        super().__init__()
        self.num_actions = num_actions
        self.mock_value = mock_value

    def _batch_size(self, state) -> int:
        if isinstance(state, MockNetworkState):
            d = state.data
            return d.shape[0] if d.dim() > 0 else 1
        if isinstance(state, torch.Tensor):
            return state.shape[0] if state.dim() > 0 else 1
        return 1

    def _make_policy_logits(self, B: int) -> torch.Tensor:
        return (
            torch.arange(self.num_actions, 0, -1, dtype=torch.float32)
            .unsqueeze(0)
            .expand(B, -1)
        )

    def _make_network_state(self, B: int) -> MockNetworkState:
        return MockNetworkState(data=torch.zeros((B, 1)))

    def obs_inference(self, obs):
        B = obs.shape[0] if isinstance(obs, torch.Tensor) and obs.dim() > 1 else 1
        return SimpleNamespace(
            value=torch.full((B,), self.mock_value, dtype=torch.float32),
            policy=torch.distributions.Categorical(
                logits=self._make_policy_logits(B)
            ),
            network_state=self._make_network_state(B),
        )

    def hidden_state_inference(self, state, action):
        B = self._batch_size(state)
        return SimpleNamespace(
            value=torch.full((B,), self.mock_value, dtype=torch.float32),
            reward=torch.zeros(B, dtype=torch.float32),
            policy=torch.distributions.Categorical(
                logits=self._make_policy_logits(B)
            ),
            network_state=self._make_network_state(B),
            to_play=torch.zeros(B, dtype=torch.int32),
        )

    def afterstate_inference(self, state, action):
        return self.hidden_state_inference(state, action)


class StateCapturingNetwork(MockSearchNetwork):
    """MockSearchNetwork that records every state passed to hidden_state_inference."""

    def __init__(self, num_actions: int, mock_value: float = 0.5):
        super().__init__(num_actions, mock_value)
        self.captured_states: list = []

    def hidden_state_inference(self, state, action):
        self.captured_states.append(state)
        return super().hidden_state_inference(state, action)


# ---------------------------------------------------------------------------
# AOS Search Mocks
# ---------------------------------------------------------------------------


class MockFlatTree:
    """Mock flat tree structure for AOS backprop tests."""

    def __init__(self, batch_size: int, num_nodes: int, num_edges: int):
        self.children_visits = torch.zeros(
            batch_size, num_nodes, num_edges, dtype=torch.int32
        )
        self.children_values = torch.zeros(
            batch_size, num_nodes, num_edges, dtype=torch.float32
        )
        self.node_visits = torch.zeros(batch_size, num_nodes, dtype=torch.int32)
        self.node_values = torch.zeros(batch_size, num_nodes, dtype=torch.float32)
        self.children_action_mask = torch.ones(
            batch_size, num_nodes, num_edges, dtype=torch.bool
        )


class MockAOSPipeline:
    """Mocks the compiled search pipeline to return fake vectorized outputs."""

    def __init__(self, batch_size: int = 2):
        self.batch_size = batch_size

    class FakeSearchOutput:
        def __init__(self, b):
            self.root_values = torch.tensor([1.0, 2.0])
            self.exploratory_policy = torch.ones((b, 2)) * 0.5
            self.target_policy = torch.ones((b, 2)) * 0.5
            self.best_actions = torch.tensor([0, 1])

    def __call__(self, obs, info, net, trajectory_actions=None):
        return self.FakeSearchOutput(self.batch_size)


# ---------------------------------------------------------------------------
# Visit Invariant MockNetwork (uniform logits, for AOS tests)
# ---------------------------------------------------------------------------


class MockAOSNetwork:
    """Simple mock network for AOS tree tests (uniform policy, unit values)."""

    def __init__(self, num_actions: int = 4):
        self.num_actions = num_actions

    def hidden_state_inference(self, state, action):
        B = action.shape[0]
        return SimpleNamespace(
            value=torch.ones(B),
            reward=torch.zeros(B),
            policy=SimpleNamespace(logits=torch.zeros((B, self.num_actions))),
            to_play=torch.zeros(B, dtype=torch.int32),
            network_state=None,
        )

    def obs_inference(self, obs):
        B = obs.shape[0]
        return SimpleNamespace(
            value=torch.zeros(B),
            policy=SimpleNamespace(logits=torch.zeros((B, self.num_actions))),
        )
