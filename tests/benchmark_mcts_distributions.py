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


def get_muzero_config():
    game_config = TicTacToeGameConfig()
    config_dict = {
        "num_simulations": 25,
        "search_batch_size": 0,  # Single simulation for benchmark
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

    start_time = time.time()
    for _ in range(num_steps):
        search_algo.run(obs, info, to_play, agent_network)
    end_time = time.time()

    duration = end_time - start_time
    fps = (num_steps * 25) / duration  # Total sims / time
    return fps, duration


class MockDist:
    def __init__(self, logits=None, probs=None):
        self.logits = logits
        self.probs = probs
        if logits is not None and probs is None:
            self.probs = torch.softmax(logits, dim=-1)
        elif probs is not None and logits is None:
            # Simple log probs for logits
            self.logits = torch.log(probs + 1e-10)

    def sample(self):
        if self.logits is not None:
            return torch.argmax(self.logits, dim=-1)
        return torch.argmax(self.probs, dim=-1)


def main():
    device = torch.device("cpu")
    config = get_muzero_config()
    input_shape = (3, 3, 3)
    num_actions = 9

    agent_network = ModularAgentNetwork(config, input_shape, num_actions)
    agent_network.eval()

    # Standard Search Algorithm
    search_algo = create_mcts(config, device, num_actions)

    print("Starting Standard MCTS Benchmark...")
    fps_std, dur_std = run_benchmark(agent_network, search_algo)
    print(f"Standard MCTS FPS: {fps_std:.2f} (Duration: {dur_std:.4f}s)")

    # --- Distribution Bypass ---

    # 1. Monkey-patch SearchAlgorithm to avoid Categorical creation
    class NoDistSearchAlgorithm(SearchAlgorithm):
        def _dist_for_batch_index(self, policy_dist, index: int):
            # Instead of returning Categorical, return a mock or just the logits
            # But the rest of the code might expect a distribution-like object
            logits = policy_dist.logits
            if logits is None:
                probs = policy_dist.probs
                logits = self._safe_log_probs(probs)

            if logits.dim() == 1:
                batch_logits = logits
            else:
                batch_logits = logits[index]

            # Return a simple object that has .logits
            return SimpleNamespace(logits=batch_logits.detach().cpu())

    # Replace Categorical call in root expansion
    original_run = SearchAlgorithm.run

    def run_no_dist(
        self, observation, info, to_play, agent_network, trajectory_action=None
    ):
        # We need to monkey-patch the Categorical call inside run
        # This is tricky because it's a local import or global in modular_search.py
        # For the benchmark, let's just use a subclass that overrides run if needed,
        # or monkey-patch the Categorical class itself in the context of the search module.
        import search.modular_search as ms

        original_categorical = ms.Categorical
        ms.Categorical = MockDist
        try:
            return original_run(
                self, observation, info, to_play, agent_network, trajectory_action
            )
        finally:
            ms.Categorical = original_categorical

    # 2. Monkey-patch PolicyHead to return MockDist
    from modules.heads.policy import PolicyHead

    original_policy_forward = PolicyHead.forward

    def policy_forward_no_dist(self, x, state=None):
        logits, new_state = super(PolicyHead, self).forward(x, state)
        # Return logits and a MockDist instead of a real one
        return logits, new_state, MockDist(logits)

    NoDistSearchAlgorithm.run = run_no_dist
    search_algo_no_dist = NoDistSearchAlgorithm(
        config,
        device,
        num_actions,
        search_algo.root_selection_strategy,
        search_algo.decision_selection_strategy,
        search_algo.chance_selection_strategy,
        search_algo.root_target_policy,
        search_algo.root_exploratory_policy,
        search_algo.prior_injectors,
        search_algo.root_searchset,
        search_algo.internal_searchset,
        search_algo.pruning_method,
        search_algo.internal_pruning_method,
        search_algo.backpropagator,
    )

    print("\nStarting Optimized (No-Dist) MCTS Benchmark...")
    # Apply monkey-patch to PolicyHead
    PolicyHead.forward = policy_forward_no_dist

    try:
        fps_opt, dur_opt = run_benchmark(agent_network, search_algo_no_dist)
    finally:
        # Restore original forward
        PolicyHead.forward = original_policy_forward

    print(f"Optimized MCTS FPS: {fps_opt:.2f} (Duration: {dur_opt:.4f}s)")

    overhead = (fps_opt - fps_std) / fps_std * 100
    print(f"\nDistribution Overhead: {overhead:.2f}%")


if __name__ == "__main__":
    main()
