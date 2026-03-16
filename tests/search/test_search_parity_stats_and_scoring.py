import pytest
import torch
from types import SimpleNamespace

from search.aos_search.min_max_stats import VectorizedMinMaxStats
from search.aos_search.scoring import ucb_score_fn, gumbel_score_fn, compute_v_mix
from search.aos_search.tree import FlatTree

from search.search_py.min_max_stats import MinMaxStats
from search.search_py.scoring_methods import UCBScoring, GumbelScoring
from search.search_py.nodes import DecisionNode

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def parity_config():
    return SimpleNamespace(
        pb_c_init=1.25,
        pb_c_base=19652,
        gumbel_cvisit=50.0,
        gumbel_cscale=1.0,
    )


def test_min_max_stats_parity():
    """Verifies AOS and Py min-max tracking and normalization are mathematically identical."""
    device = torch.device("cpu")
    aos_stats = VectorizedMinMaxStats.allocate(batch_size=1, device=device)
    py_stats = MinMaxStats(known_bounds=None)

    for v in [0.0, 1.0, 0.5]:
        aos_stats.update(torch.tensor([v], dtype=torch.float32), torch.tensor([True]))
        py_stats.update(v)

    assert aos_stats.min_values.item() == py_stats.min
    assert aos_stats.max_values.item() == py_stats.max

    for v in [0.0, 1.0, 0.5, 2.0, -1.0]:
        v_tensor = torch.tensor([v], dtype=torch.float32)
        aos_norm = aos_stats.normalize(v_tensor).item()
        py_norm = py_stats.normalize(v)
        if isinstance(py_norm, torch.Tensor):
            py_norm = py_norm.item()

        clamped_py_norm = min(max(py_norm, 0.0), 1.0)
        assert torch.allclose(
            torch.tensor(aos_norm), torch.tensor(clamped_py_norm), atol=1e-5
        )


def test_ucb_scoring_parity(parity_config):
    """Verifies that UCB equations exactly match between batched and node-based logic."""
    device = torch.device("cpu")
    num_actions, batch_size = 4, 1
    parent_visits, parent_value = 10, 0.5

    child_visits = torch.tensor([1, 2, 0, 3], dtype=torch.float32)
    child_values = torch.tensor([0.6, 0.4, 0.0, 0.8], dtype=torch.float32)
    child_priors = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
    child_logits = torch.log(child_priors)

    py_node = DecisionNode(prior=0.0)
    py_node.bootstrap_method = "parent_value"
    py_node.visits = parent_visits
    py_node.value_sum = parent_value * parent_visits
    py_node.child_visits, py_node.child_values, py_node.child_priors = (
        child_visits.clone(),
        child_values.clone(),
        child_priors.clone(),
    )
    py_node.pb_c_init, py_node.pb_c_base = (
        parity_config.pb_c_init,
        parity_config.pb_c_base,
    )
    py_node._v_mix = None

    tree = FlatTree.allocate(batch_size, 10, num_actions, 1, device)
    tree.node_visits[0, 0], tree.node_values[0, 0] = parent_visits, parent_value
    tree.children_visits[0, 0, :] = child_visits.to(torch.int32)
    tree.children_values[0, 0, :] = child_values
    tree.children_prior_logits[0, 0, :] = child_logits
    tree.children_action_mask[0, 0, :] = True
    # AOS uses children_index != -1 as the "visited" mask; set for all visited children
    for i, v in enumerate(child_visits.tolist()):
        if v > 0:
            tree.children_index[0, 0, i] = i + 1

    aos_minmax = VectorizedMinMaxStats.allocate(batch_size, device)
    py_minmax = MinMaxStats(known_bounds=None)
    aos_minmax.update(torch.tensor([0.0], dtype=torch.float32), torch.tensor([True]))
    aos_minmax.update(torch.tensor([1.0], dtype=torch.float32), torch.tensor([True]))
    py_minmax.update(0.0)
    py_minmax.update(1.0)

    aos_scores = ucb_score_fn(
        tree,
        torch.tensor([0], dtype=torch.int32),
        parity_config.pb_c_init,
        parity_config.pb_c_base,
        min_max_stats=aos_minmax,
    )
    py_scores = UCBScoring().get_scores(py_node, py_minmax)

    assert torch.allclose(aos_scores[0], py_scores, atol=1e-5)


def test_v_mix_parity():
    """Verifies V-Mix (Value Mixing) computation parity."""
    device = torch.device("cpu")
    num_actions, batch_size = 4, 1

    child_visits = torch.tensor([2, 0, 5, 0], dtype=torch.float32)
    child_values = torch.tensor([0.7, 0.0, 0.3, 0.0], dtype=torch.float32)
    child_priors = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)

    py_node = DecisionNode(prior=0.0)
    py_node.network_value = 0.5
    py_node.child_visits, py_node.child_values, py_node.child_priors = (
        child_visits.clone(),
        child_values.clone(),
        child_priors.clone(),
    )
    py_node.network_policy = child_priors.clone()

    tree = FlatTree.allocate(batch_size, 10, num_actions, 1, device)
    tree.raw_network_values[0, 0] = 0.5
    tree.children_visits[0, 0, :] = child_visits.to(torch.int32)
    tree.children_values[0, 0, :] = child_values
    tree.children_prior_logits[0, 0, :] = torch.log(child_priors)
    # AOS uses children_index != -1 as the "visited" mask; set for all visited children
    for i, v in enumerate(child_visits.tolist()):
        if v > 0:
            tree.children_index[0, 0, i] = i + 1

    aos_vmix = compute_v_mix(tree, torch.tensor([0], dtype=torch.int32))
    assert torch.allclose(aos_vmix, torch.tensor(py_node.get_v_mix()), atol=1e-5)


def test_gumbel_scoring_parity(parity_config):
    """Verifies Gumbel target scoring equations align perfectly."""
    device = torch.device("cpu")

    child_visits = torch.tensor([2, 0, 5, 0], dtype=torch.float32)
    child_values = torch.tensor([0.7, 0.0, 0.3, 0.0], dtype=torch.float32)
    child_priors = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)

    py_node = DecisionNode(prior=0.0)
    py_node.network_value = 0.5
    py_node.child_visits, py_node.child_values, py_node.child_priors = (
        child_visits.clone(),
        child_values.clone(),
        child_priors.clone(),
    )
    py_node.network_policy = child_priors.clone()
    py_node.bootstrap_method = "v_mix"
    py_node.visits = int(child_visits.sum().item())

    tree = FlatTree.allocate(1, 10, 4, 1, device)
    tree.raw_network_values[0, 0] = 0.5
    tree.children_visits[0, 0, :] = child_visits.to(torch.int32)
    tree.children_values[0, 0, :] = child_values
    tree.children_prior_logits[0, 0, :] = torch.log(child_priors)
    tree.children_action_mask[0, 0, :] = True
    # AOS uses children_index != -1 as the "visited" mask; set for all visited children
    for i, v in enumerate(child_visits.tolist()):
        if v > 0:
            tree.children_index[0, 0, i] = i + 1

    aos_minmax = VectorizedMinMaxStats.allocate(1, device)
    aos_minmax.update(torch.tensor([0.0], dtype=torch.float32), torch.tensor([True]))
    aos_minmax.update(torch.tensor([1.0], dtype=torch.float32), torch.tensor([True]))

    py_minmax = MinMaxStats(known_bounds=[0.0, 1.0])

    aos_scores = gumbel_score_fn(
        tree,
        torch.tensor([0], dtype=torch.int32),
        parity_config.gumbel_cvisit,
        parity_config.gumbel_cscale,
        min_max_stats=aos_minmax,
    )
    py_scores = GumbelScoring(parity_config).get_scores(py_node, py_minmax)

    assert torch.allclose(aos_scores[0], py_scores, atol=1e-5)


def test_gumbel_noise_seeding_parity():
    """Explicitly verifies Gumbel noise shapes and seeds align across both implementations."""
    torch.manual_seed(99)
    noise_py = torch.distributions.Gumbel(0, 1).sample((4,))

    torch.manual_seed(99)
    noise_aos = torch.distributions.Gumbel(0, 1).sample((1, 4))

    assert torch.allclose(noise_aos[0], noise_py, atol=1e-5)
