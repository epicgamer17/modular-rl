import time
import torch
import numpy as np
import os

from configs.games.tictactoe_config import TicTacToeConfig
from configs.agents.muzero import MuZeroConfig
from agents.policies.search_policy import SearchPolicy
from search.search_factories import create_mcts
from modules.world_models.muzero_world_model import MuzeroWorldModel
from modules.agent_nets.muzero import Network


def test_memory_growth(num_calls=1000, num_simulations=25):
    print(
        f"Testing for memory growth and FPS stability ({num_calls} calls, {num_simulations} sims)..."
    )

    game_config = TicTacToeConfig()
    params = {
        "num_simulations": num_simulations,
        "dense_layer_widths": [64],
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
        "lstm_hidden_size": 64,
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

    latencies = []

    # Store all info_dicts to simulate replay buffer growth (if they were leaky, memory would explode)
    all_info_dicts = []

    for i in range(num_calls):
        start = time.perf_counter()
        action = policy.compute_action(obs, info)
        latency = time.perf_counter() - start
        latencies.append(latency)

        info_dict = policy.get_info()
        all_info_dicts.append(info_dict)

        if (i + 1) % 100 == 0:
            avg_latency = np.mean(latencies[-100:])
            print(
                f"  Call {i+1}/{num_calls}: Avg Latency: {avg_latency:.4f}s (FPS: {1.0/avg_latency:.2f})"
            )

    first_100 = np.mean(latencies[:100])
    last_100 = np.mean(latencies[-100:])

    print("-" * 30)
    print(f"First 100 calls avg: {first_100:.4f}s")
    print(f"Last 100 calls avg: {last_100:.4f}s")

    # Check if last 100 is significantly slower than first 100 (e.g. > 20% slower)
    if last_100 > first_100 * 1.2:
        print("WARNING: Significant performance degradation detected!")
        sys.exit(1)
    else:
        print("SUCCESS: Performance remained stable.")

    # Check for grad_fn in stored tensors (safeguard)
    for info_dict in all_info_dicts:
        policy_tensor = info_dict["policy"]
        if policy_tensor.grad_fn is not None:
            print("ERROR: Stored policy tensor has grad_fn!")
            sys.exit(1)

        metadata = info_dict["search_metadata"]

        # Recursively check metadata for grad_fn
        def check_leaks(obj):
            if isinstance(obj, torch.Tensor):
                if obj.grad_fn is not None:
                    return True
                if (
                    obj.is_leaf is False and obj.requires_grad
                ):  # Tensors with grad might not have grad_fn if in no_grad but still keep graph
                    return True
            elif isinstance(obj, dict):
                return any(check_leaks(v) for v in obj.values())
            elif isinstance(obj, list):
                return any(check_leaks(v) for v in obj)
            return False

        if check_leaks(metadata):
            print("ERROR: Stored search metadata contains tensors with grad/graph!")
            sys.exit(1)

    print("SUCCESS: No leaky tensors detected in stored metadata.")
    env.close()


if __name__ == "__main__":
    test_memory_growth()
