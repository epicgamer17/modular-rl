import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

import importlib
import time

import numpy as np
import torch


def _import_legacy_memory_growth_stack():
    required = [
        ("configs.games.tictactoe", "TicTacToeConfig"),
        ("agents.policies.search_policy", "SearchPolicy"),
        ("search.search_factories", "create_mcts"),
        ("modules.models.world_model", "WorldModel"),
        ("modules.models.agent_network", "AgentNetwork"),
        ("configs.agents.muzero", "MuZeroConfig"),
    ]

    resolved = {}
    for mod_name, symbol in required:
        try:
            mod = importlib.import_module(mod_name)
            resolved[symbol] = getattr(mod, symbol)
        except Exception as exc:
            pytest.skip(
                f"Legacy memory-growth harness unavailable ({mod_name}.{symbol}): {exc}",
                allow_module_level=False,
            )

    return resolved


def test_memory_growth(num_calls=100, num_simulations=10):
    stack = _import_legacy_memory_growth_stack()
    TicTacToeConfig = stack["TicTacToeConfig"]
    SearchPolicy = stack["SearchPolicy"]
    create_mcts = stack["create_mcts"]
    WorldModel = stack["WorldModel"]
    AgentNetwork = stack["AgentNetwork"]
    MuZeroConfig = stack["MuZeroConfig"]

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
        "world_model_cls": WorldModel,
        "lstm_hidden_size": 64,
        "bootstrap_method": "v_mix",
        "support_range": None,
        "stochastic": True,
        "use_value_prefix": False,
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
        "action_selector": {"base": {"type": "argmax", "kwargs": {}}},
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

    model = AgentNetwork(config, num_actions, obs_shape, world_model_cls=WorldModel).to(
        device
    )
    search_algo = create_mcts(config, device, num_actions)
    policy = SearchPolicy(model, search_algo, config, device, simple_obs_shape)

    latencies = []
    for _ in range(num_calls):
        start = time.perf_counter()
        policy.compute_action(obs, info)
        latencies.append(time.perf_counter() - start)

    first_chunk = np.mean(latencies[:20])
    last_chunk = np.mean(latencies[-20:])
    assert last_chunk <= first_chunk * 1.5

    env.close()
