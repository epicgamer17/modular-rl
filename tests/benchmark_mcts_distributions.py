import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn as nn
from types import SimpleNamespace
from search.modular_search import SearchAlgorithm
from modules.agent_nets.modular import ModularAgentNetwork
from configs.agents.muzero import MuZeroConfig
from search.search_selectors import SelectionStrategy
from search.backpropogation import Backpropagator
from search.search_factories import create_mcts
from search.nodes import DecisionNode, ChanceNode
from configs.games.game import GameConfig


# Dummy Game Config for TicTacToe
class TicTacToeGameConfig(GameConfig):
    def __init__(self):
        super().__init__(
            max_score=1.0,
            min_score=-1.0,
            is_discrete=True,
            is_image=True,
            is_deterministic=True,
            has_legal_moves=True,
            perfect_information=True,
            multi_agent=True,
            num_players=2,
            num_actions=9,
            make_env=lambda: None,
        )


def get_muzero_config(search_batch_size=0):
    game_config = TicTacToeGameConfig()
    config_dict = {
        "num_simulations": 25,
        "search_batch_size": search_batch_size,
        "discount_factor": 0.99,
        "pb_c_init": 1.25,
        "pb_c_base": 19652,
        "q_estimation_method": "v_mix",
        "known_bounds": None,
        "soft_update": False,
        "min_max_epsilon": 0.0,
        "gumbel": False,
        "stochastic": False,
        "value_prefix": False,
        "lstm_hidden_size": 16,
        "lstm_horizon_len": 5,
        "representation_backbone": {"type": "dense", "hidden_widths": [16]},
        "dynamics_backbone": {"type": "dense", "hidden_widths": [16]},
        "prediction_backbone": {"type": "dense", "hidden_widths": [16]},
        "reward_head": {"neck": {"type": "dense", "hidden_widths": [16]}},
        "value_head": {"neck": {"type": "dense", "hidden_widths": [16]}},
        "policy_head": {"neck": {"type": "dense", "hidden_widths": [16]}},
        "action_selector": {"base": {"type": "mcts"}},
        "optimizer": {"adam": {"lr": 1e-4}},
        "replay_buffer": {"type": "modular", "capacity": 1000},
    }
    return MuZeroConfig(config_dict, game_config)


def run_benchmark(agent_network, search_algo, num_steps=50):
    obs = torch.zeros(3, 3, 3)
    info = {"legal_moves": [list(range(9))]}
    to_play = 0

    # Warmup
    for _ in range(5):
        search_algo.run(obs, info, to_play, agent_network)

    start_time = time.time()
    for _ in range(num_steps):
        search_algo.run(obs, info, to_play, agent_network)
    end_time = time.time()

    duration = end_time - start_time
    # Total simulations = num_steps * num_simulations
    fps = (num_steps * 25) / duration
    return fps, duration


class MockDist:
    def __init__(self, logits=None, probs=None):
        self.logits = logits
        self.probs = probs
        if logits is not None and probs is None:
            self.probs = torch.softmax(logits, dim=-1)
        elif probs is not None and logits is None:
            self.logits = torch.log(probs + 1e-10)

    def sample(self):
        if self.logits is not None:
            return torch.argmax(self.logits, dim=-1)
        return torch.argmax(self.probs, dim=-1)


# Distribution Bypass Components
class NoDistSearchAlgorithm(SearchAlgorithm):
    def _dist_for_batch_index(self, policy_dist, index: int):
        logits = policy_dist.logits
        if logits is None:
            probs = policy_dist.probs
            logits = self._safe_log_probs(probs)

        if logits.dim() == 1:
            batch_logits = logits
        else:
            batch_logits = logits[index]

        return SimpleNamespace(logits=batch_logits.detach().cpu())


original_run = SearchAlgorithm.run


def run_no_dist(
    self, observation, info, to_play, agent_network, trajectory_action=None
):
    import search.modular_search as ms

    original_categorical = ms.Categorical
    ms.Categorical = MockDist
    try:
        return original_run(
            self, observation, info, to_play, agent_network, trajectory_action
        )
    finally:
        ms.Categorical = original_categorical


def run_test_case(agent_network, batch_size, use_dist, num_steps=50):
    config = get_muzero_config(search_batch_size=batch_size)
    device = agent_network.device
    num_actions = agent_network.num_actions

    std_algo = create_mcts(config, device, num_actions)

    # Debugging unbatching
    original_unbatch = agent_network.unbatch_network_states

    def debug_unbatch(batched_state):
        res = original_unbatch(batched_state)
        # if isinstance(batched_state, MuZeroNetworkState):
        #     print(f"DEBUG: Unbatching MuZeroNetworkState. Batch size detected: {len(res)}")
        return res

    agent_network.unbatch_network_states = debug_unbatch

    if use_dist:
        search_algo = std_algo
    else:
        search_algo = NoDistSearchAlgorithm(
            config,
            device,
            num_actions,
            std_algo.root_selection_strategy,
            std_algo.decision_selection_strategy,
            std_algo.chance_selection_strategy,
            std_algo.root_target_policy,
            std_algo.root_exploratory_policy,
            std_algo.prior_injectors,
            std_algo.root_searchset,
            std_algo.internal_searchset,
            std_algo.pruning_method,
            std_algo.internal_pruning_method,
            std_algo.backpropagator,
        )
        # Use our patched run
        search_algo.run = run_no_dist.__get__(search_algo, NoDistSearchAlgorithm)

    # Monkey-patch PolicyHead if using bypass
    from modules.heads.policy import PolicyHead

    original_policy_forward = PolicyHead.forward

    if not use_dist:

        def policy_forward_no_dist(self, x, state=None):
            logits, new_state = super(PolicyHead, self).forward(x, state)
            return logits, new_state, MockDist(logits)

        PolicyHead.forward = policy_forward_no_dist

    try:
        sps, duration = run_benchmark(agent_network, search_algo, num_steps)
    finally:
        PolicyHead.forward = original_policy_forward

    return sps, duration


def main():
    device = torch.device("cpu")
    # Base config for network initialization
    base_config = get_muzero_config()
    input_shape = (3, 3, 3)
    num_actions = 9

    agent_network = ModularAgentNetwork(base_config, input_shape, num_actions)
    agent_network.eval()

    test_cases = [
        {"batch_size": 0, "use_dist": True, "label": "Sequential, With Dist"},
        {"batch_size": 0, "use_dist": False, "label": "Sequential, No Dist"},
        {"batch_size": 5, "use_dist": True, "label": "Batched (5), With Dist"},
        {"batch_size": 5, "use_dist": False, "label": "Batched (5), No Dist"},
    ]

    print(f"{'Configuration':<30} | {'Sims/sec':<10} | {'Duration':<10}")
    print("-" * 55)

    results = {}
    for case in test_cases:
        label = case["label"]
        sps, duration = run_test_case(
            agent_network, case["batch_size"], case["use_dist"]
        )
        results[label] = sps
        print(f"{label:<30} | {sps:<10.2f} | {duration:<10.4f}s")

    # Analysis
    print("\nImpact of Distribution Creation:")
    seq_overhead = (
        (results["Sequential, No Dist"] - results["Sequential, With Dist"])
        / results["Sequential, With Dist"]
        * 100
    )
    batch_overhead = (
        (results["Batched (5), No Dist"] - results["Batched (5), With Dist"])
        / results["Batched (5), With Dist"]
        * 100
    )
    print(f"  Sequential Overhead: {seq_overhead:.2f}%")
    print(f"  Batched Overhead:    {batch_overhead:.2f}%")

    print("\nImpact of Batching:")
    dist_speedup = (
        (results["Batched (5), With Dist"] - results["Sequential, With Dist"])
        / results["Sequential, With Dist"]
        * 100
    )
    nodist_speedup = (
        (results["Batched (5), No Dist"] - results["Sequential, No Dist"])
        / results["Sequential, No Dist"]
        * 100
    )
    print(f"  Batching Speedup (With Dist): {dist_speedup:.2f}%")
    print(f"  Batching Speedup (No Dist):   {nodist_speedup:.2f}%")


if __name__ == "__main__":
    main()
