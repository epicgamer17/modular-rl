import sys
import os
from tests.test_search_parity_standalone import get_config, MockNetwork
import torch


def debug_gumbel_q():
    config = get_config()
    device = torch.device("cpu")
    num_actions = 4
    batch_size = 1
    config.num_simulations = 20
    config.max_nodes = 100
    config.gumbel = True
    config.scoring_method = "gumbel"
    config.policy_extraction = "gumbel"
    config.use_sequential_halving = True
    config.gumbel_m = 2
    config.q_estimation_method = "v_mix"
    config.estimation_method = "v_mix"
    config.value_prefix = False
    net = MockNetwork(num_actions)
    obs = torch.ones((batch_size, 1, 4, 4))
    info = {"legal_moves": [list(range(num_actions))]}
    py_info = {"legal_moves": info["legal_moves"][0]}

    print("=== AOS ===")
    torch.manual_seed(42)
    from search.aos_search.search_factories import build_search_pipeline
    import search.aos_search.search_factories

    run_mcts = build_search_pipeline(config, device, num_actions)

    # Let's intercept to print tree internals
    original_gumbel = search.aos_search.search_factories.gumbel_max_q_policy

    def patched_gumbel(tree, **kwargs):
        print(f"AOS Q: {tree.children_values[0, 0, :].tolist()}")
        print(f"AOS N: {tree.children_visits[0, 0, :].tolist()}")
        print(f"AOS P: {tree.children_prior_logits[0, 0, :].tolist()}")
        return original_gumbel(tree, **kwargs)

    search.aos_search.search_factories.gumbel_max_q_policy = patched_gumbel

    aos_output = run_mcts(obs, info, net)

    print("=== PY ===")
    torch.manual_seed(42)
    from search.search_py.search_factories import create_mcts
    import search.search_py.modular_search

    # Intercept run to print py root
    original_run = search.search_py.modular_search.SearchAlgorithm.run

    def patched_run(self, obs, info, to_play, net, **kwargs):
        v, e, t, b, meta = original_run(self, obs, info, to_play, net, **kwargs)
        root = self.root
        print(
            f"PY Q: {[root.children[a].value() if a in root.children else 0.0 for a in range(4)]}"
        )
        print(
            f"PY N: {[root.children[a].visits if a in root.children else 0 for a in range(4)]}"
        )
        import math

        print(
            f"PY P: {[math.log(root.children[a].prior) if a in root.children else -float('inf') for a in range(4)]}"
        )
        print(f"PY v_mix: {root.get_v_mix()}")
        return v, e, t, b, meta

    search.search_py.modular_search.SearchAlgorithm.run = patched_run

    mcts_py = create_mcts(config, device, num_actions)

    # Reset seed before PySearch
    import numpy as np

    torch.manual_seed(42)
    np.random.seed(42)

    py_val, py_expl, py_target, py_best, py_meta = mcts_py.run(obs[0], py_info, 0, net)

    aos_output = run_mcts(obs, info, net)

    print(f"AOS Root Values: {aos_output.tree.children_values[0, 0]}")
    print(f"AOS Root Visits: {aos_output.tree.children_visits[0, 0]}")

    print(
        f"PY Root Values: {[c.value() if c else 0.0 for c in mcts_py.root.children.values()]}"
    )
    print(
        f"PY Root Visits: {[c.visit_count if c else 0 for c in mcts_py.root.children.values()]}"
    )

debug_gumbel_q()
```
