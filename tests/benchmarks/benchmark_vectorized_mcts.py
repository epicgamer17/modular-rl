import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn as nn
from search.modular_search import ModularSearch
from modules.agent_nets.modular import ModularAgentNetwork
from configs.agents.muzero import MuZeroConfig
from configs.games.game import GameConfig
from search.search_factories import create_mcts


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
        "bootstrap_method": "v_mix",
        "known_bounds": None,
        "soft_update": False,
        "min_max_epsilon": 0.0,
        "gumbel": False,
        "stochastic": False,
        "use_value_prefix": False,
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


def run_benchmark_vectorized(agent_network, search_algo, env_batch_size, num_steps=50):
    batched_obs = torch.zeros(env_batch_size, 3, 3, 3)
    batched_info = {"legal_moves": [[list(range(9))] for _ in range(env_batch_size)]}
    batched_to_play = [0] * env_batch_size

    # Warmup
    for _ in range(2):
        search_algo.run_vectorized(
            batched_obs, batched_info, batched_to_play, agent_network
        )

    start_time = time.time()
    for _ in range(num_steps):
        search_algo.run_vectorized(
            batched_obs, batched_info, batched_to_play, agent_network
        )
    end_time = time.time()

    duration = end_time - start_time
    # Total simulations = num_steps * env_batch_size * num_simulations
    fps = (num_steps * env_batch_size * 25) / duration
    return fps, duration


def main():
    device = torch.device("cpu")
    base_config = get_muzero_config()
    input_shape = (3, 3, 3)
    num_actions = 9

    agent_network = ModularAgentNetwork(base_config, input_shape, num_actions)
    agent_network.eval()

    test_cases = [
        {"env_batch_size": 1, "search_batch_size": 0, "label": "Env_1_Search_0"},
        {"env_batch_size": 1, "search_batch_size": 5, "label": "Env_1_Search_5"},
        {"env_batch_size": 16, "search_batch_size": 0, "label": "Env_16_Search_0"},
        {"env_batch_size": 16, "search_batch_size": 5, "label": "Env_16_Search_5"},
    ]

    print(f"{'Configuration':<25} | {'Sims/sec':<10} | {'Duration':<10}")
    print("-" * 50)

    for case in test_cases:
        config = get_muzero_config(search_batch_size=case["search_batch_size"])
        search_algo = create_mcts(config, device, num_actions)
        sps, duration = run_benchmark_vectorized(
            agent_network, search_algo, case["env_batch_size"]
        )
        print(f"{case['label']:<25} | {sps:<10.2f} | {duration:<10.4f}s")


if __name__ == "__main__":
    main()
