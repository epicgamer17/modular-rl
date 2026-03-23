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


def test_memory_growth(net_factory, num_calls=100, num_simulations=10):
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
        "unroll_steps": 5,
        "discount_factor": 1.0,
        "stochastic": True,
        "num_chance": 10,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "checkpoint_interval": 1000,
        "pb_c_init": 1.25,
        "pb_c_base": 19652,
        "gumbel": False,
        "arch": {"backbone": {"type": "dense", "hidden_dim": 64}},
        "prediction_backbone": {"type": "dense", "hidden_dim": 64},
        "world_model": {"latent_dim": 64},
        "heads": {
            "state_value": {"output_strategy": {"type": "scalar"}},
            "policy": {"output_strategy": {"type": "categorical"}},
            "reward": {"output_strategy": {"type": "scalar"}},
        },
        "agent_type": "muzero",
        "action_selector": {"base": {"type": "argmax", "kwargs": {}}},
    }

    config = MuZeroConfig(params, game_config)
    device = torch.device("cpu")

    env = game_config.env_factory()
    env.reset()
    obs = env.observe(env.agent_selection)
    info = env.infos[env.agent_selection]
    simple_obs_shape = obs.shape
    obs_shape = (1, *obs.shape)
    num_actions = env.action_space(env.possible_agents[0]).n

    model = net_factory(config, obs_shape, num_actions).to(device)
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
