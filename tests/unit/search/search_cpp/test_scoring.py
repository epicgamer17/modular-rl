import math
import pytest
import torch

# Try to import the C++ backend. Skip if not available.
try:
    import search

    search.set_backend("cpp")
    import search.search_cpp as search_cpp
except (ImportError, RuntimeError) as e:
    pytest.skip(f"search_cpp backend not available: {e}", allow_module_level=True)

pytestmark = pytest.mark.unit


def test_search_cpp_puct_scoring_math_verification():
    """
    Verify the Upper Confidence Bound calculation using the MuZero PUCT formula
    for the search_cpp backend.

    Setup:
    - Node visit count N(s) = 10
    - Action A: Q = 0.5, P = 0.3, N(s,a) = 5
    - Action B: Q = 0.2, P = 0.7, N(s,a) = 0

    Expected math (hand-calculated):
    - expected_score_a = 1.19773083
    - expected_score_b = 2.76823163
    """
    arena = search_cpp.NodeArena()
    node_idx = arena.create_decision(prior=1.0)
    node = arena.decision(node_idx)

    # Expand node with 2 children
    # expand(to_play, network_policy, priors, allowed_actions, reward, network_value)
    node.expand(
        to_play=0,
        network_policy=[0.3, 0.7],
        priors=[0.3, 0.7],
        allowed_actions=[0, 1],
        reward=0.0,
        network_value=0.2,
    )

    # Manually set child stats to match the test case
    # In search_cpp, child stats are std::vector exposed as numpy-like views or lists?
    # Actually, from bindings: child_visits, child_values, child_priors are properties.
    # But they might be read-only in the bindings. Let's check.
    # bindings.cpp: .def_property_readonly("child_visits", &Node::child_visits)
    # Wait, if they are read-only, I might need to simulate visits to set them.

    # Let's check search_cpp.NodeArena for any 'update' method or just use backprop.
    # Alternatively, if I can't set them directly, I might have to use backprop.

    # Actually, let's see if I can use a simpler approach if they are read-only.
    # If they are read-only, I can't easily "mock" them without doing full simulations.

    # Wait, the C++ Node class might have setters for these in the C++ side but perhaps not exposed.
    # Let's check Node in bindings.cpp again.
    # 234:       .def_property_readonly("child_priors", &Node::child_priors)
    # 235:       .def_property_readonly("child_values", &Node::child_values)
    # 236:       .def_property_readonly("child_visits", &Node::child_visits)

    # Okay, they ARE read-only.
    # To get Action A to have N=5, Q=0.5, I need to backpropagate 5 times.

    # But wait, there's another way: search_cpp.compute_scores takes an arena and node_index.
    # If I can't set the node stats, I'll have to use backprop.

    # Let's try to set them anyway, maybe pybind11 allows it if there's a setter I missed.
    # No, they are explicitly readonly.

    # Okay, I'll use backprop to set the values.
    # Initial: N=10, V=0.35
    # For A: N=5, Q=0.5. Total sum = 2.5
    # For B: N=0, Q=0.2. Total sum = 0.0

    # Wait, if I want N(s)=10 total, and Action A has 5, where are the other 5?
    # Maybe 5 other visits to some other actions? Or just parent visits = 10 but children sum to 5?
    # The scoring formula uses node.visits() for N(s).

    # Let's manually backprop.
    config = search_cpp.BackpropConfig()
    config.discount_factor = 1.0
    config.num_players = 1

    bp = search_cpp.AverageDiscountedReturnBackpropagator()
    min_max_stats = search_cpp.MinMaxStats()

    # Action A: 5 visits with value 0.5
    # Action B: 0 visits
    # Total parent visits should be 10.
    # So 5 visits to A, and 5 visits to some other (or just set parent visits directly if possible).

    # Node visits is also readonly: .def_property("visits", &Node::visits, &Node::set_visits)
    # Ah! Visits HAS a setter: .set_visits.
    # Value sum also has a setter: .set_value_sum.
    # But child_visits does NOT.

    # Wait, I can create children and backprop through them.
    child_a_idx = arena.create_decision(prior=0.3, parent_index=node_idx)
    node.set_child(0, child_a_idx)

    # We want Action A to have N=5, Q=0.5.
    # Total parent visits should be 10.
    # So we'll do 5 backprops for A, and 5 for a dummy child action 2.
    child_dummy_idx = arena.create_decision(prior=0.0, parent_index=node_idx)
    # CRITICAL: Re-fetch node after create_decision to avoid invalid reference due to vector reallocation
    node = arena.decision(node_idx)
    node.set_child(2, child_dummy_idx)

    # Perform 5 backprops for Action A with value 0.5
    for _ in range(5):
        bp.backpropagate(
            arena=arena,
            search_path=[node_idx, child_a_idx],
            action_path=[0],
            leaf_value=0.5,
            leaf_to_play=0,
            min_max_stats=min_max_stats,
            config=config,
        )

    # Perform 5 backprops for dummy action
    for _ in range(5):
        bp.backpropagate(
            arena=arena,
            search_path=[node_idx, child_dummy_idx],
            action_path=[2],
            leaf_value=0.2,  # matching B
            leaf_to_play=0,
            min_max_stats=min_max_stats,
            config=config,
        )

    # Re-fetch node again for final updates
    node = arena.decision(node_idx)
    # Now node.visits=10, child_visits[0]=5, child_visits[1]=0.
    # Check parent value (it will be (5*0.5 + 5*0.2)/10 = 0.35)
    # We want it to be 0.2 for B's bootstrap.
    # So let's override parent value_sum to get 0.2.
    node.value_sum = 2.0

    # Setup MinMaxStats with [0.2, 0.5]
    mm = search_cpp.MinMaxStats()
    mm.update(0.2)
    mm.update(0.5)

    scoring_cfg = search_cpp.ScoringConfig()
    scoring_cfg.pb_c_init = 1.25
    scoring_cfg.pb_c_base = 19652.0

    # Compute scores
    scores = search_cpp.compute_scores(
        search_cpp.ScoringMethodType.UCB, arena, node_idx, mm, scoring_cfg
    )

    score_a = scores[0]
    score_b = scores[1]

    expected_score_a = 1.19773083
    expected_score_b = 2.76823163

    assert math.isclose(
        score_a, expected_score_a, rel_tol=1e-7
    ), f"Action A score {score_a} != {expected_score_a}"
    assert math.isclose(
        score_b, expected_score_b, rel_tol=1e-7
    ), f"Action B score {score_b} != {expected_score_b}"
    assert score_b > score_a
