import time
import torch
import numpy as np
import os
import sys

# Ensure custom_gym_envs_pkg is in path
sys.path.append(os.path.abspath("custom_gym_envs_pkg"))

from configs.games.catan_config import CatanConfig
from configs.games.tictactoe_config import TicTacToeConfig
from configs.agents.muzero import MuZeroConfig
from agents.policies.search_policy import SearchPolicy
from search.search_factories import create_mcts
from modules.world_models.muzero_world_model import MuzeroWorldModel
from modules.agent_nets.muzero import Network


def benchmark_env(game_name, make_env_fn, num_steps=100):
    print(f"Benchmarking Environment: {game_name}")
    env = make_env_fn()

    # reset benchmark
    start = time.perf_counter()
    for _ in range(10):
        env.reset()
    reset_time = (time.perf_counter() - start) / 10
    print(f"  Average Reset Time: {reset_time:.4f}s")

    # step benchmark
    env.reset()
    start = time.perf_counter()
    steps_taken = 0
    for _ in range(num_steps):
        agent = env.agent_selection
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            env.reset()
            continue

        mask = info.get("action_mask")
        if mask is not None:
            legal_actions = np.where(mask)[0]
            action = np.random.choice(legal_actions)
        else:
            action = env.action_space(agent).sample()

        env.step(action)
        steps_taken += 1

    step_time = (time.perf_counter() - start) / steps_taken
    print(f"  Average Step Time: {step_time:.4f}s")
    env.close()
    return reset_time, step_time


def benchmark_policy(game_name, game_config, num_simulations=25, num_calls=10):
    print(f"Benchmarking Policy: {game_name} (sims={num_simulations})")

    params = {
        "num_simulations": num_simulations,
        "dense_layer_widths": [64, 64, 64],
        "residual_layers": [],
        "conv_layers": [],
        "representation_residual_layers": [],
        "representation_conv_layers": [],
        "dynamics_residual_layers": [],
        "dynamics_conv_layers": [],
        "reward_conv_layers": [],
        "reward_dense_layer_widths": [16],
        "to_play_conv_layers": [],
        "to_play_dense_layer_widths": [16],
        "critic_conv_layers": [],
        "critic_dense_layer_widths": [16],
        "actor_conv_layers": [],
        "actor_dense_layer_widths": [16],
        "chance_conv_layers": [],
        "chance_dense_layer_widths": [16],
        "world_model_cls": MuzeroWorldModel,
        "lstm_hidden_size": 64,  # Matches notebook
        "q_estimation_method": "v_mix",
        "support_range": None,
        "stochastic": True,
        "value_prefix": False,
        "pb_c_init": 1.25,
        "pb_c_base": 19652,
        "gumbel": False,
        "gumbel_m": 8,
        "gumbel_cvisit": 50,
        "gumbel_cscale": 1.0,
        "discount_factor": 1.0,
        "soft_update": False,
        "min_max_epsilon": 0.01,
        "search_batch_size": 0,
    }

    config = MuZeroConfig(params, game_config)
    device = torch.device("cpu")

    env = game_config.make_env()
    env.reset()
    obs = env.observe(env.agent_selection)
    info = env.infos[env.agent_selection]
    simple_obs_shape = obs.shape
    obs_shape = (1, *obs.shape)
    num_actions = env.action_space(env.possible_agents[0]).n

    model = Network(
        config, num_actions, obs_shape, world_model_cls=MuzeroWorldModel
    ).to(device)
    search_algo = create_mcts(config, device, num_actions)
    policy = SearchPolicy(model, search_algo, config, device, simple_obs_shape)

    # Warmup
    policy.compute_action(obs, info)

    start = time.perf_counter()
    for _ in range(num_calls):
        policy.compute_action(obs, info)
    policy_time = (time.perf_counter() - start) / num_calls

    print(f"  Average Policy Predict Time: {policy_time:.4f}s")
    print(f"  Estimated FPS: {1.0/policy_time:.2f}")

    env.close()
    return policy_time


if __name__ == "__main__":
    # 1. Environment Benchmarks
    catan_reset, catan_step = benchmark_env("Catan", CatanConfig().make_env)
    ttt_reset, ttt_step = benchmark_env("TicTacToe", TicTacToeConfig().make_env)

    print("-" * 30)
    print(f"Environments: Catan is {catan_step/ttt_step:.1f}x slower per step")
    print("-" * 30)

    # 2. Policy Benchmarks
    catan_policy = benchmark_policy("Catan", CatanConfig())
    ttt_policy = benchmark_policy("TicTacToe", TicTacToeConfig())

    print("-" * 30)
    print(f"Policy: Catan is {catan_policy/ttt_policy:.1f}x slower per call")
    print("-" * 30)
