import sys
import os
from tests.test_search_parity_standalone import get_config, MockNetwork
import torch


def debug_ucb():
    config = get_config()
    device = torch.device("cpu")
    num_actions = 4
    batch_size = 1
    config.num_simulations = 1
    config.max_nodes = 100
    config.gumbel = False
    config.scoring_method = "ucb"
    config.policy_extraction = "visit_count"
    config.q_estimation_method = "mcts_value"
    config.estimation_method = "mcts_value"
    config.value_prefix = False
    config.discount_factor = 0.99
    net = MockNetwork(num_actions)

    # Let's wrap inference to print the value
    original_hidden = net.hidden_state_inference

    def wrapped_hidden(state, action):
        out = original_hidden(state, action)
        print(
            f"MockNetwork HIDDEN state={state}, action={action} => val={out.value.item()}, reward={out.reward.mean().item()}"
        )
        return out

    net.hidden_state_inference = wrapped_hidden

    obs = torch.ones((batch_size, 1, 4, 4))
    info = {"legal_moves": [list(range(num_actions))]}
    py_info = {"legal_moves": info["legal_moves"][0]}

    # Print BP again
    from search.aos_search.backpropogation import average_discounted_backprop

    original_bp = average_discounted_backprop

    def printed_bp(
        tree, batch_idx, nodes_at_d, actions_at_d, current_values, discount, valid_mask
    ):
        print(
            f"AOS BP: depth node={nodes_at_d[0].item()}, action={actions_at_d[0].item()}, curr_val={current_values[0].item()}, discount={discount}"
        )
        return original_bp(
            tree,
            batch_idx,
            nodes_at_d,
            actions_at_d,
            current_values,
            discount,
            valid_mask,
        )

    import search.aos_search.search_factories

    search.aos_search.search_factories._BACKPROP_REGISTRY["average"] = printed_bp

    print("=== AOS ===")
    torch.manual_seed(42)
    run_mcts = search.aos_search.search_factories.build_search_pipeline(
        config, device, num_actions
    )
    aos_output = run_mcts(obs, info, net)
    print(f"aos_root_values: {aos_output.root_values[0]}")


debug_ucb()
