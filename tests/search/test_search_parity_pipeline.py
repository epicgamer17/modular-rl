import pytest
import torch
import numpy as np
from types import SimpleNamespace

from agents.factories.aos_search import build_search_pipeline
from agents.factories.search_py import create_mcts
from tests.search.conftest import MockSearchNetwork as MockNetwork

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def mcts_config():
    """Generates a base config suitable for both factories."""
    return SimpleNamespace(
        pb_c_init=1.25,
        pb_c_base=19652,
        discount_factor=0.99,
        gumbel=False,
        gumbel_cvisit=50.0,
        gumbel_cscale=1.0,
        game=SimpleNamespace(num_players=1),
        use_value_prefix=False,
        num_simulations=4,
        num_codes=1,
        max_search_depth=50,
        max_nodes=100,
        use_dirichlet=False,
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
        root_dirichlet_alpha_adaptive=True,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        injection_frac=0.1,
        policy_extraction="visit_count",
        backprop_method="average",
        scoring_method="ucb",
        known_bounds=[0.0, 2.0],
        use_sequential_halving=False,
        gumbel_m=2,
        bootstrap_method="network_value",
        soft_update=False,
        min_max_epsilon=1e-8,
        stochastic=False,
        search_batch_size=0,
        virtual_loss=1.0,
        use_virtual_mean=False,
        compilation=SimpleNamespace(enabled=False, fullgraph=False),
        internal_decision_modifier="none",
        internal_chance_modifier="none",
        stochastic_exploration=False,
        sampling_temp=1.0,
    )


def _run_parity_check(config, net, obs, info, py_info, num_actions, seed=42):
    """Helper to execute and assert both pipelines deterministically."""
    device = torch.device("cpu")
    run_mcts_aos = build_search_pipeline(config, device, num_actions)
    mcts_py = create_mcts(config, device, num_actions)

    torch.manual_seed(seed)
    np.random.seed(seed)
    aos_info = {**info, "player": torch.tensor([0], dtype=torch.int8)}
    aos_output = run_mcts_aos(obs, aos_info, net)

    torch.manual_seed(seed)
    np.random.seed(seed)
    py_info_with_player = {**py_info, "player": 0}
    _, _, py_target, _, _ = mcts_py.run(obs[0], py_info_with_player, net)

    assert torch.allclose(aos_output.target_policy[0], py_target, atol=1e-5)


def test_full_search_ucb_parity(mcts_config, monkeypatch):
    """Standard UCB rollout parity."""
    monkeypatch.setattr("search.nodes.DecisionNode.bootstrap_method", "parent_value")
    num_actions, batch_size = 4, 1

    net = MockNetwork(num_actions)
    obs = torch.ones((batch_size, 1, 4, 4))
    info = {"legal_moves": [list(range(num_actions))]}

    _run_parity_check(
        mcts_config,
        net,
        obs,
        info,
        {"legal_moves": info["legal_moves"][0]},
        num_actions,
    )


def test_full_search_gumbel_parity(mcts_config, monkeypatch):
    """Gumbel sequential halving parity."""
    monkeypatch.setattr("search.nodes.DecisionNode.bootstrap_method", "parent_value")
    mcts_config.gumbel = True
    mcts_config.scoring_method = "gumbel"
    mcts_config.policy_extraction = "gumbel"
    mcts_config.use_sequential_halving = True
    num_actions = 4

    net = MockNetwork(num_actions)
    obs = torch.ones((1, 1, 4, 4))
    info = {"legal_moves": [list(range(num_actions))]}

    _run_parity_check(
        mcts_config,
        net,
        obs,
        info,
        {"legal_moves": info["legal_moves"][0]},
        num_actions,
    )


def test_variable_batch_sizes(mcts_config, monkeypatch):
    """Verifies AOS batched output at index 0 strictly matches Python's single-item execution."""
    monkeypatch.setattr("search.nodes.DecisionNode.bootstrap_method", "parent_value")
    num_actions, batch_size = 4, 5

    net = MockNetwork(num_actions)
    obs = torch.ones((batch_size, 1, 4, 4))
    info = {"legal_moves": [list(range(num_actions)) for _ in range(batch_size)]}

    _run_parity_check(
        mcts_config,
        net,
        obs,
        info,
        {"legal_moves": info["legal_moves"][0]},
        num_actions,
        seed=123,
    )


def test_extreme_value_clipping_parity(mcts_config, monkeypatch):
    """Ensures massive values (+1e6) behave identically in MinMax scaling."""
    monkeypatch.setattr("search.nodes.DecisionNode.bootstrap_method", "parent_value")
    mcts_config.known_bounds = None
    num_actions = 4

    net = MockNetwork(num_actions, mock_value=1e6)
    obs = torch.ones((1, 1, 4, 4))
    info = {"legal_moves": [list(range(num_actions))]}

    _run_parity_check(
        mcts_config,
        net,
        obs,
        info,
        {"legal_moves": info["legal_moves"][0]},
        num_actions,
        seed=77,
    )


def test_deep_horizon_discounting(mcts_config, monkeypatch):
    """Forces deep rollouts to verify geometric discounting parity down the tree."""
    monkeypatch.setattr("search.nodes.DecisionNode.bootstrap_method", "parent_value")
    mcts_config.num_simulations = 30
    mcts_config.max_search_depth = 100
    num_actions = 4

    net = MockNetwork(num_actions)
    obs = torch.ones((1, 1, 4, 4))
    info = {"legal_moves": [list(range(num_actions))]}

    _run_parity_check(
        mcts_config,
        net,
        obs,
        info,
        {"legal_moves": info["legal_moves"][0]},
        num_actions,
        seed=55,
    )
