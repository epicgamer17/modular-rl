import sys
import os
from tests.test_search_parity_standalone import get_config, MockNetwork
import torch


def debug_gumbel_halving():
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

    # 1. Patch AOS Halving
    from search.aos_search.dynamic_masking import apply_sequential_halving

    original_halving = apply_sequential_halving

    def patched_halving(tree, current_sim_idx, total_simulations, base_m, **kwargs):
        original_halving(tree, current_sim_idx, total_simulations, base_m, **kwargs)
        mask = tree.children_action_mask[0, 0, :].cpu()
        print(f"AOS Sim {current_sim_idx}: mask={mask}")

    import search.aos_search.batched_mcts

    # Need to patch where it is USED in batched_mcts
    original_batched_mcts = search.aos_search.search_factories.build_search_pipeline
    import search.aos_search.search_factories

    search.aos_search.search_factories.apply_sequential_halving = patched_halving

    print("=== AOS ===")
    torch.manual_seed(42)
    run_mcts = search.aos_search.search_factories.build_search_pipeline(
        config, device, num_actions
    )
    aos_output = run_mcts(obs, info, net)

    # 2. Patch PY Halving
    from search.search_py.pruners import SequentialHalvingPruning

    original_py_halving = SequentialHalvingPruning.step

    def patched_py_halving(self, node, state, config, min_max_stats, sim_index):
        searchset, state = original_py_halving(
            self, node, state, config, min_max_stats, sim_index
        )
        if hasattr(node, "parent") and node.parent is None:
            acts = searchset.actions if searchset else "All"
            print(f"PY Sim {sim_index}: Searchset={acts}")
        return searchset, state

    SequentialHalvingPruning.step = patched_py_halving

    print("=== PY ===")
    torch.manual_seed(42)
    from search.search_py.search_factories import create_mcts

    mcts_py = create_mcts(config, device, num_actions)
    print(f"PySearch pruning method: {type(mcts_py.pruning_method)}")
    py_val, py_expl, py_target, py_best, py_meta = mcts_py.run(obs[0], py_info, 0, net)


debug_gumbel_halving()
