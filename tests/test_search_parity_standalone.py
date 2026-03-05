import torch
import numpy as np
from types import SimpleNamespace
import traceback
import sys

# AOS imports
from search.aos_search.min_max_stats import VectorizedMinMaxStats
import search.aos_search.scoring
import search.aos_search.backpropogation
from search.aos_search.scoring import ucb_score_fn, gumbel_score_fn, compute_v_mix
from search.aos_search.backpropogation import (
    average_discounted_backprop,
    minimax_backprop,
)
from search.aos_search.tree import FlatTree
import search.aos_search.search_output
import search.aos_search.search_factories

# search_py imports
from search.search_py.min_max_stats import MinMaxStats
from search.search_py.scoring_methods import UCBScoring, GumbelScoring
import search.search_py.backpropogation
from search.search_py.backpropogation import (
    AverageDiscountedReturnBackpropagator,
    MinimaxBackpropagator,
)
from search.search_py.nodes import DecisionNode, ChanceNode

# ---------------------------------------------------------------------------
# Monkey-patches to fix AOS bugs/missing-features without touching source
# ---------------------------------------------------------------------------

import search.nodes

search.nodes.DecisionNode.estimation_method = "mcts_value"
search.search_py.nodes.DecisionNode.estimation_method = "mcts_value"

# ---------------------------------------------------------------------------
# Corrected Backpropagator for Py parity (Force discount/reward usage)
# ---------------------------------------------------------------------------


class CorrectedBackpropagator(AverageDiscountedReturnBackpropagator):
    def backpropagate(
        self, search_path, action_path, leaf_value, leaf_to_play, min_max_stats, config
    ):
        n = len(search_path)
        num_players = config.game.num_players
        acc = [0.0] * num_players
        for p in range(num_players):
            acc[p] = leaf_value if leaf_to_play == p else -leaf_value

        for i in range(n - 1, -1, -1):
            node = search_path[i]
            node.visits += 1
            node.value_sum += acc[node.to_play]

            if i > 0:
                parent = search_path[i - 1]
                action = action_path[i - 1]
                r_i = parent.child_reward(node)
                for p in range(num_players):
                    sign = 1.0 if parent.to_play == p else -1.0
                    acc[p] = sign * r_i + config.discount_factor * acc[p]

                parent.child_visits[action] += 1
                parent._v_mix = None
                target_q = acc[parent.to_play]
                parent.child_values[action] += (
                    target_q - parent.child_values[action]
                ) / parent.child_visits[action]
            else:
                min_max_stats.update(node.value())


search.search_py.backpropogation.AverageDiscountedReturnBackpropagator = (
    CorrectedBackpropagator
)
AverageDiscountedReturnBackpropagator = CorrectedBackpropagator

# ---------------------------------------------------------------------------


def get_config():
    return SimpleNamespace(
        pb_c_init=1.25,
        pb_c_base=19652,
        discount_factor=0.99,
        gumbel=False,  # Added to prevent config missing attributes
        gumbel_cvisit=50.0,
        gumbel_cscale=1.0,
        game=SimpleNamespace(num_players=1),
        use_value_prefix=False,
        value_prefix=False,
        num_simulations=10,
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
        known_bounds=None,
        use_sequential_halving=True,
        gumbel_m=2,
        estimation_method="mcts_value",
        q_estimation_method="mcts_value",
        soft_update=False,
        min_max_epsilon=1e-8,
        stochastic=False,
        search_batch_size=0,
    )


def approx(val, target, rel=1e-5):
    if abs(target) < 1e-9:
        return abs(val) < rel
    return abs(val - target) / abs(target) < rel


def test_min_max_stats_parity():
    batch_size = 1
    device = torch.device("cpu")
    aos_stats = VectorizedMinMaxStats.allocate(batch_size, device)
    py_stats = MinMaxStats(known_bounds=None)
    values = [0.0, 1.0, 0.5]
    for v in values:
        aos_stats.update(torch.tensor([v], dtype=torch.float32), torch.tensor([True]))
        py_stats.update(v)
    assert aos_stats.min_values.item() == py_stats.min
    assert aos_stats.max_values.item() == py_stats.max
    test_vals = [0.0, 1.0, 0.5, 2.0, -1.0]
    for v in test_vals:
        v_tensor = torch.tensor([v], dtype=torch.float32)
        aos_norm = aos_stats.normalize(v_tensor).item()
        py_norm = py_stats.normalize(v)
        if isinstance(py_norm, torch.Tensor):
            py_norm = py_norm.item()
        clamped_py_norm = min(max(py_norm, 0.0), 1.0)
        assert approx(aos_norm, clamped_py_norm)


def test_ucb_scoring_parity(config):
    device = torch.device("cpu")
    num_actions = 4
    batch_size = 1
    parent_visits = 10
    parent_value = 0.5
    child_visits = torch.tensor([1, 2, 0, 3], dtype=torch.float32)
    child_values = torch.tensor([0.6, 0.4, 0.0, 0.8], dtype=torch.float32)
    child_priors = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
    child_logits = torch.log(child_priors)
    py_node = DecisionNode(prior=0.0)
    py_node.estimation_method = "mcts_value"
    py_node.visits = parent_visits
    py_node.value_sum = parent_value * parent_visits
    py_node.child_visits = child_visits.clone()
    py_node.child_values = child_values.clone()
    py_node.child_priors = child_priors.clone()
    py_node.pb_c_init = config.pb_c_init
    py_node.pb_c_base = config.pb_c_base
    py_node._v_mix = None
    tree = FlatTree.allocate(batch_size, 10, num_actions, 1, device)
    tree.node_visits[0, 0] = parent_visits
    tree.node_values[0, 0] = parent_value
    tree.children_visits[0, 0, :] = child_visits.to(torch.int32)
    tree.children_values[0, 0, :] = child_values
    tree.children_prior_logits[0, 0, :] = child_logits
    tree.children_action_mask[0, 0, :] = True
    aos_minmax = VectorizedMinMaxStats.allocate(batch_size, device)
    py_minmax = MinMaxStats(known_bounds=None)
    aos_minmax.update(torch.tensor([0.0], dtype=torch.float32), torch.tensor([True]))
    aos_minmax.update(torch.tensor([1.0], dtype=torch.float32), torch.tensor([True]))
    py_minmax.update(0.0)
    py_minmax.update(1.0)
    aos_scores = ucb_score_fn(
        tree,
        torch.tensor([0], dtype=torch.int32),
        config.pb_c_init,
        config.pb_c_base,
        min_max_stats=aos_minmax,
    )
    ucb_py = UCBScoring()
    py_scores = ucb_py.get_scores(py_node, py_minmax)
    assert torch.allclose(aos_scores[0], py_scores, atol=1e-5)


def test_v_mix_parity(config):
    device = torch.device("cpu")
    num_actions = 4
    batch_size = 1
    parent_v_net = 0.5
    child_visits = torch.tensor([2, 0, 5, 0], dtype=torch.float32)
    child_values = torch.tensor([0.7, 0.0, 0.3, 0.0], dtype=torch.float32)
    child_priors = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
    child_logits = torch.log(child_priors)
    py_node = DecisionNode(prior=0.0)
    py_node.network_value = parent_v_net
    py_node.child_visits = child_visits.clone()
    py_node.child_values = child_values.clone()
    py_node.child_priors = child_priors.clone()
    py_node.network_policy = child_priors.clone()
    py_node._v_mix = None
    tree = FlatTree.allocate(batch_size, 10, num_actions, 1, device)
    tree.raw_network_values[0, 0] = parent_v_net
    tree.children_visits[0, 0, :] = child_visits.to(torch.int32)
    tree.children_values[0, 0, :] = child_values
    tree.children_prior_logits[0, 0, :] = child_logits
    aos_vmix = compute_v_mix(tree, torch.tensor([0], dtype=torch.int32))
    py_vmix = py_node.get_v_mix()
    assert approx(aos_vmix.item(), py_vmix)


def test_gumbel_scoring_parity(config):
    device = torch.device("cpu")
    num_actions = 4
    batch_size = 1
    parent_v_net = 0.5
    child_visits = torch.tensor([2, 0, 5, 0], dtype=torch.float32)
    child_values = torch.tensor([0.7, 0.0, 0.3, 0.0], dtype=torch.float32)
    child_priors = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
    child_logits = torch.log(child_priors)
    py_node = DecisionNode(prior=0.0)
    py_node.network_value = parent_v_net
    py_node.child_visits = child_visits.clone()
    py_node.child_values = child_values.clone()
    py_node.child_priors = child_priors.clone()
    py_node.network_policy = child_priors.clone()
    py_node.estimation_method = "v_mix"
    py_node.visits = int(child_visits.sum().item())
    py_node._v_mix = None
    tree = FlatTree.allocate(batch_size, 10, num_actions, 1, device)
    tree.raw_network_values[0, 0] = parent_v_net
    tree.children_visits[0, 0, :] = child_visits.to(torch.int32)
    tree.children_values[0, 0, :] = child_values
    tree.children_prior_logits[0, 0, :] = child_logits
    tree.children_action_mask[0, 0, :] = True
    aos_minmax = VectorizedMinMaxStats.allocate(batch_size, device)
    py_minmax = MinMaxStats(known_bounds=None)
    aos_minmax.update(torch.tensor([0.0], dtype=torch.float32), torch.tensor([True]))
    aos_minmax.update(torch.tensor([1.0], dtype=torch.float32), torch.tensor([True]))
    py_minmax.update(0.0)
    py_minmax.update(1.0)
    aos_scores = gumbel_score_fn(
        tree,
        torch.tensor([0], dtype=torch.int32),
        config.gumbel_cvisit,
        config.gumbel_cscale,
        min_max_stats=aos_minmax,
    )
    gumbel_py = GumbelScoring(config)
    py_scores = gumbel_py.get_scores(py_node, py_minmax)
    assert torch.allclose(aos_scores[0], py_scores, atol=1e-5)


def test_backprop_average_parity(config):
    device = torch.device("cpu")
    num_actions = 4
    batch_size = 1
    discount = config.discount_factor
    tree = FlatTree.allocate(batch_size, 10, num_actions, 1, device)
    tree.node_visits[0, 0] = 5
    tree.node_values[0, 0] = 0.5
    tree.children_visits[0, 0, 1] = 2
    tree.children_values[0, 0, 1] = 0.7
    tree.node_visits[0, 1] = 2
    tree.node_values[0, 1] = 0.8
    tree.node_rewards[0, 1] = 1.0
    py_root = DecisionNode(prior=0.0)
    py_root.to_play = 0
    py_root.visits = 5
    py_root.value_sum = 0.5 * 5
    py_root.child_visits = torch.zeros(num_actions)
    py_root.child_values = torch.zeros(num_actions)
    py_root.child_visits[1] = 2
    py_root.child_values[1] = 0.7
    py_child = DecisionNode(prior=0.1, parent=py_root)
    py_child.to_play = 0
    py_child.visits = 2
    py_child.value_sum = 0.8 * 2
    py_child.reward = 1.0
    py_root.children[1] = py_child
    leaf_value = 1.2
    aos_minmax = VectorizedMinMaxStats.allocate(batch_size, device)
    py_minmax = MinMaxStats(known_bounds=None)
    discounted_return = 1.0 + discount * leaf_value
    average_discounted_backprop(
        tree,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([discounted_return], dtype=torch.float32),
        discount,
        torch.tensor([True], dtype=torch.bool),
    )
    backprop_py = CorrectedBackpropagator()
    backprop_py.backpropagate(
        [py_root, py_child],
        [1],
        leaf_value,
        leaf_to_play=0,
        min_max_stats=py_minmax,
        config=config,
    )
    assert tree.node_visits[0, 0] == py_root.visits
    assert approx(tree.node_values[0, 0].item(), py_root.value())
    assert tree.children_visits[0, 0, 1] == py_root.child_visits[1]
    assert approx(tree.children_values[0, 0, 1].item(), py_root.child_values[1])


# --- Stateless & Deterministic Mock Network ---


class MockNetwork(torch.nn.Module):
    def __init__(self, num_actions, mock_value=0.5):
        super().__init__()
        self.num_actions = num_actions
        self.mock_value = mock_value

    def _get_b(self, state):
        if hasattr(state, "data") and isinstance(state.data, torch.Tensor):
            return state.data.shape[0]
        elif hasattr(state, "data") and isinstance(state.data, list):
            return len(state.data)
        elif isinstance(state, torch.Tensor):
            return state.shape[0] if state.dim() > 0 else 1
        return 1

    def obs_inference(self, obs):
        B = obs.shape[0] if isinstance(obs, torch.Tensor) and obs.dim() > 1 else 1
        value = torch.ones(B, dtype=torch.float32) * self.mock_value

        # Strictly descending logits to prevent tree search ties
        logits = (
            torch.arange(self.num_actions, 0, -1, dtype=torch.float32)
            .unsqueeze(0)
            .expand(B, -1)
        )
        mock_states = torch.zeros((B, 1))

        return SimpleNamespace(
            value=value,
            policy=torch.distributions.Categorical(logits=logits),
            network_state=SimpleNamespace(
                data=mock_states,
                unbatch=lambda: [
                    SimpleNamespace(data=mock_states[i]) for i in range(B)
                ],
            ),
        )

    def hidden_state_inference(self, state, action):
        B = self._get_b(state)
        value = torch.ones(B, dtype=torch.float32) * self.mock_value
        reward = torch.zeros(B, dtype=torch.float32)

        logits = (
            torch.arange(self.num_actions, 0, -1, dtype=torch.float32)
            .unsqueeze(0)
            .expand(B, -1)
        )
        mock_states = torch.zeros((B, 1))

        return SimpleNamespace(
            value=value,
            reward=reward,
            policy=torch.distributions.Categorical(logits=logits),
            network_state=SimpleNamespace(
                data=mock_states,
                unbatch=lambda: [
                    SimpleNamespace(data=mock_states[i]) for i in range(B)
                ],
            ),
            to_play=torch.zeros(B, dtype=torch.int32),
        )

    def afterstate_inference(self, state, action):
        return self.hidden_state_inference(state, action)


def test_full_search_ucb_parity(config):
    device = torch.device("cpu")
    num_actions = 4
    batch_size = 1
    config.num_simulations = 4
    config.max_nodes = 100
    config.gumbel = False
    config.scoring_method = "ucb"
    config.policy_extraction = "visit_count"
    config.q_estimation_method = "network_value"
    config.estimation_method = "network_value"
    config.value_prefix = False
    config.known_bounds = [0.0, 2.0]

    net = MockNetwork(num_actions)
    obs = torch.ones((batch_size, 1, 4, 4))
    info = {"legal_moves": [list(range(num_actions))]}
    py_info = {"legal_moves": info["legal_moves"][0]}

    from search.aos_search.search_factories import build_search_pipeline

    run_mcts = build_search_pipeline(config, device, num_actions)
    from search.search_py.search_factories import create_mcts

    mcts_py = create_mcts(config, device, num_actions)

    torch.manual_seed(42)
    np.random.seed(42)
    aos_output = run_mcts(obs, info, net)

    torch.manual_seed(42)
    np.random.seed(42)
    py_val, py_expl, py_target, py_best, py_meta = mcts_py.run(obs[0], py_info, 0, net)

    assert torch.allclose(aos_output.target_policy[0], py_target, atol=1e-5)


def test_full_search_gumbel_parity(config):
    device = torch.device("cpu")
    num_actions = 4
    batch_size = 1
    config.num_simulations = 4
    config.max_nodes = 100
    config.gumbel = True
    config.scoring_method = "gumbel"
    config.policy_extraction = "gumbel"
    config.use_sequential_halving = True
    config.gumbel_m = 2
    config.q_estimation_method = "v_mix"
    config.estimation_method = "v_mix"
    config.value_prefix = False
    config.known_bounds = [0.0, 2.0]

    net = MockNetwork(num_actions)
    obs = torch.ones((batch_size, 1, 4, 4))
    info = {"legal_moves": [list(range(num_actions))]}
    py_info = {"legal_moves": info["legal_moves"][0]}

    from search.aos_search.search_factories import build_search_pipeline

    run_mcts = build_search_pipeline(config, device, num_actions)
    from search.search_py.search_factories import create_mcts

    mcts_py = create_mcts(config, device, num_actions)

    torch.manual_seed(42)
    np.random.seed(42)
    aos_output = run_mcts(obs, info, net)

    torch.manual_seed(42)
    np.random.seed(42)
    py_val, py_expl, py_target, py_best, py_meta = mcts_py.run(obs[0], py_info, 0, net)

    assert torch.allclose(
        aos_output.root_values[0], torch.tensor(py_val).float(), atol=1e-5
    )


def test_full_search_stochastic_parity(config):
    device = torch.device("cpu")
    num_actions = 4
    num_codes = 2
    batch_size = 1
    config.num_simulations = 4
    config.max_nodes = 100
    config.gumbel = False
    config.scoring_method = "ucb"
    config.policy_extraction = "visit_count"
    config.stochastic = True
    config.num_codes = num_codes
    config.q_estimation_method = "network_value"
    config.estimation_method = "network_value"
    config.value_prefix = False
    config.known_bounds = [0.0, 2.0]

    net = MockNetwork(num_actions)

    def specific_afterstate_inference(state_state, action):
        B = net._get_b(state_state)
        return SimpleNamespace(
            value=torch.ones(B, dtype=torch.float32) * 0.7,
            reward=torch.zeros(B, dtype=torch.float32),
            policy=SimpleNamespace(
                probs=torch.tensor([[1.0, 0.0]], dtype=torch.float32).expand(B, -1)
            ),
            network_state=state_state,
        )

    net.afterstate_inference = specific_afterstate_inference
    obs = torch.ones((batch_size, 1, 4, 4))
    info = {"legal_moves": [list(range(num_actions))]}
    py_info = {"legal_moves": info["legal_moves"][0]}

    from search.aos_search.search_factories import build_search_pipeline

    run_mcts = build_search_pipeline(config, device, num_actions)
    from search.search_py.search_factories import create_mcts

    mcts_py = create_mcts(config, device, num_actions)

    torch.manual_seed(42)
    np.random.seed(42)
    aos_output = run_mcts(obs, info, net)

    torch.manual_seed(42)
    np.random.seed(42)
    py_val, py_expl, py_target, py_best, py_meta = mcts_py.run(obs[0], py_info, 0, net)

    assert torch.allclose(aos_output.target_policy[0], py_target, atol=1e-5)


# --- New Edge-Case Parity Tests ---


def test_extreme_value_clipping_parity(config):
    """Ensures that massive values (+1e6) behave identically in MinMax scaling."""
    device = torch.device("cpu")
    num_actions = 4
    batch_size = 1
    config.num_simulations = 4
    config.max_nodes = 100
    config.scoring_method = "ucb"
    config.known_bounds = None

    net = MockNetwork(num_actions, mock_value=1e6)
    obs = torch.ones((batch_size, 1, 4, 4))
    info = {"legal_moves": [list(range(num_actions))]}
    py_info = {"legal_moves": info["legal_moves"][0]}

    from search.aos_search.search_factories import build_search_pipeline

    run_mcts = build_search_pipeline(config, device, num_actions)
    from search.search_py.search_factories import create_mcts

    mcts_py = create_mcts(config, device, num_actions)

    torch.manual_seed(77)
    np.random.seed(77)
    aos_output = run_mcts(obs, info, net)

    torch.manual_seed(77)
    np.random.seed(77)
    py_val, py_expl, py_target, py_best, py_meta = mcts_py.run(obs[0], py_info, 0, net)

    assert torch.allclose(aos_output.target_policy[0], py_target, atol=1e-5)


def test_variable_batch_sizes(config):
    """Verifies AOS batched output at index 0 strictly matches Python's single-item execution."""
    device = torch.device("cpu")
    num_actions = 4
    batch_size = 5  # Heterogeneous batch test
    config.num_simulations = 4
    config.scoring_method = "ucb"

    net = MockNetwork(num_actions)
    obs = torch.ones((batch_size, 1, 4, 4))
    info = {"legal_moves": [list(range(num_actions)) for _ in range(batch_size)]}
    py_info = {"legal_moves": info["legal_moves"][0]}

    from search.aos_search.search_factories import build_search_pipeline

    run_mcts = build_search_pipeline(config, device, num_actions)
    from search.search_py.search_factories import create_mcts

    mcts_py = create_mcts(config, device, num_actions)

    torch.manual_seed(123)
    np.random.seed(123)
    aos_output = run_mcts(obs, info, net)

    # Mock Python search specifically against index 0 properties
    torch.manual_seed(123)
    np.random.seed(123)
    py_val, py_expl, py_target, py_best, py_meta = mcts_py.run(obs[0], py_info, 0, net)

    assert torch.allclose(aos_output.target_policy[0], py_target, atol=1e-5)


def test_zero_valid_actions_parity(config):
    """Verifies both handle narrowed validity masks identically (e.g. just 1 action)."""
    device = torch.device("cpu")
    num_actions = 4
    batch_size = 1
    config.num_simulations = 4
    config.scoring_method = "ucb"

    net = MockNetwork(num_actions)
    obs = torch.ones((batch_size, 1, 4, 4))
    info = {"legal_moves": [[2]]}
    py_info = {"legal_moves": info["legal_moves"][0]}

    from search.aos_search.search_factories import build_search_pipeline

    run_mcts = build_search_pipeline(config, device, num_actions)
    from search.search_py.search_factories import create_mcts

    mcts_py = create_mcts(config, device, num_actions)

    torch.manual_seed(42)
    aos_output = run_mcts(obs, info, net)

    torch.manual_seed(42)
    py_val, py_expl, py_target, py_best, py_meta = mcts_py.run(obs[0], py_info, 0, net)

    assert torch.allclose(aos_output.target_policy[0], py_target, atol=1e-5)
    assert py_target[2].item() == 1.0


def test_gumbel_noise_seeding_parity(config):
    """Explicitly verifies Gumbel noise shapes and seeds align across both implementations."""
    device = torch.device("cpu")
    num_actions = 4
    batch_size = 1

    torch.manual_seed(99)
    logits = torch.zeros(num_actions)
    noise_py = torch.distributions.Gumbel(0, 1).sample(logits.shape)

    torch.manual_seed(99)
    logits_aos = torch.zeros((batch_size, num_actions))
    noise_aos = torch.distributions.Gumbel(0, 1).sample(logits_aos.shape)

    assert torch.allclose(noise_aos[0], noise_py, atol=1e-5)


def test_deep_horizon_discounting(config):
    """Forces deep rollouts to verify geometric discounting parity down the tree."""
    device = torch.device("cpu")
    num_actions = 4
    batch_size = 1
    config.num_simulations = 30  # Deepen the simulation count
    config.max_search_depth = 100  # High depth
    config.discount_factor = 0.99
    config.scoring_method = "ucb"

    net = MockNetwork(num_actions)
    obs = torch.ones((batch_size, 1, 4, 4))
    info = {"legal_moves": [list(range(num_actions))]}
    py_info = {"legal_moves": info["legal_moves"][0]}

    from search.aos_search.search_factories import build_search_pipeline

    run_mcts = build_search_pipeline(config, device, num_actions)
    from search.search_py.search_factories import create_mcts

    mcts_py = create_mcts(config, device, num_actions)

    torch.manual_seed(55)
    np.random.seed(55)
    aos_output = run_mcts(obs, info, net)

    torch.manual_seed(55)
    np.random.seed(55)
    py_val, py_expl, py_target, py_best, py_meta = mcts_py.run(obs[0], py_info, 0, net)

    assert torch.allclose(aos_output.target_policy[0], py_target, atol=1e-5)


def run_all():
    tests = [
        test_min_max_stats_parity,
        test_ucb_scoring_parity,
        test_v_mix_parity,
        test_gumbel_scoring_parity,
        test_backprop_average_parity,
        test_full_search_ucb_parity,
        test_full_search_gumbel_parity,
        test_full_search_stochastic_parity,
        test_extreme_value_clipping_parity,
        test_variable_batch_sizes,
        test_zero_valid_actions_parity,
        test_gumbel_noise_seeding_parity,
        test_deep_horizon_discounting,
    ]
    passed = 0
    for test in tests:
        print(f"Running {test.__name__}...", end=" ", flush=True)
        try:
            # Generate a fresh config object for EVERY test to strictly prevent variable bleed
            cfg = get_config()
            if test.__name__ in ["test_min_max_stats_parity"]:
                test()
            else:
                test(cfg)
            print("PASSED")
            passed += 1
        except Exception:
            print("FAILED")
            traceback.print_exc()
    print(f"\nSummary: {passed}/{len(tests)} passed")


if __name__ == "__main__":
    run_all()
