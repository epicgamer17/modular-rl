import pytest

pytestmark = pytest.mark.unit

import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from search.search_py.modular_search import ModularSearch

# Sync imports with modular_search.py to ensure same class objects
from search.nodes import DecisionNode, ChanceNode
from search.min_max_stats import MinMaxStats
from modules.models.inference_output import InferenceOutput
from configs.agents.muzero import MuZeroConfig
from configs.games.tictactoe import TicTacToeConfig


from modules.models.agent_network import AgentNetwork


@pytest.fixture
def search_setup(net_factory):
    def _setup(stochastic=False):
        game_config = TicTacToeConfig()
        num_actions = game_config.num_actions

        # Minimal MuZero Config with all required fields
        config_dict = {
            "action_selector": {"base": {"type": "mcts"}},
            "training_steps": 1000,
            "minibatch_size": 64,
            "replay_buffer_size": 5000,
            "unroll_steps": 5,
            "results_path": "results",
            "num_simulations": 10,
            "discount_factor": 0.99,
            "pb_c_init": 1.25,
            "pb_c_base": 19652,
            "gumbel": False,
            "gumbel_m": 1,
            "virtual_loss": 3.0,
            "use_virtual_mean": False,
            "bootstrap_method": "v_mix",
            "scoring_method": "ucb",
            "max_search_depth": 5,
            "known_bounds": None,
            "min_max_epsilon": 1e-8,
            "stochastic": stochastic,
            "num_chance": 2 if stochastic else 1,
            "norm_type": "none",
            "learning_rate": 0.001,
            "optimizer": "adam",
            "adam_epsilon": 1e-8,
            "momentum": 0.9,
            "weight_decay": 0.0,
            "lr_schedule": {"type": "constant"},
            "test_interval": 1000,
            "checkpoint_interval": 1000,
            "n_step": 1,
            "per_alpha": 0.5,
            "per_beta_schedule": {"type": "constant"},
            "per_epsilon": 1e-6,
            "per_use_batch_weights": False,
            "per_use_initial_max_priority": True,
            "bootstrap_on_truncated": False,
            "record_video": False,
            "record_video_interval": 1000,
            "executor_type": "local",
            "num_workers": 1,
            "num_envs_per_worker": 1,
            "atom_size": 1,
            "support_range": None,
            "games_per_generation": 100,
            "value_loss_factor": 1.0,
            "to_play_loss_factor": 1.0,
            "temperature_schedule": {"type": "constant", "initial": 1.0},
            "clip_low_prob": 0.0,
            "value_loss_function": "mse",
            "reward_loss_function": "mse",
            "policy_loss_function": "cross_entropy",
            "to_play_loss_function": "cross_entropy",
            "reanalyze_ratio": 0.0,
            "reanalyze_method": "mcts",
            "reanalyze_tau": 0.3,
            "injection_frac": 0.0,
            "reanalyze_noise": False,
            "reanalyze_update_priorities": False,
            "consistency_loss_factor": 0.0,
            "projector_output_dim": 128,
            "projector_hidden_dim": 128,
            "predictor_output_dim": 128,
            "predictor_hidden_dim": 64,
            "action_embedding_dim": 32,
            "single_action_plane": False,
            "latent_viz_method": "pca",
            "latent_viz_interval": 1,
            "noisy_sigma": 0.0,
            "use_value_prefix": False,
            "lstm_horizon_len": 5,
            "lstm_hidden_size": 16,
            "compilation": {"mode": "reduce-overhead"},
            "arch": {"backbone": {"type": "dense", "widths": [32, 32]}},
            "prediction_backbone": {"type": "dense", "widths": [32]},
            "value_head": {
                "output_strategy": {"type": "scalar"},
                "neck": {"type": "dense", "widths": [16]},
            },
            "policy_head": {
                "output_strategy": {"type": "categorical", "num_classes": num_actions},
                "neck": {"type": "dense", "widths": [16]},
            },
            "reward_head": {
                "output_strategy": {"type": "scalar"},
                "neck": {"type": "dense", "widths": [16]},
            },
            "to_play_head": {
                "output_strategy": {
                    "type": "categorical",
                    "num_classes": game_config.num_players,
                },
                "neck": {"type": "dense", "widths": [16]},
            },
            "chance_probability_head": {
                "type": "categorical",
                "num_classes": 2 if stochastic else 1,
                "neck": {"type": "dense", "widths": [16]},
            },
        }

        config = MuZeroConfig(config_dict, game_config)
        device = torch.device("cpu")

        from search.search_py.search_factories import create_mcts

        search_engine = create_mcts(config, device, num_actions)

        # CRITICAL: Set node configs so pb_c values are populated
        search_engine._set_node_configs()

        # Use real lightweight network for compliance
        env = game_config.make_env()
        input_shape = env.observation_space(env.possible_agents[0]).shape
        network = net_factory(config, input_shape, num_actions)

        env.reset()
        obs = env.observe(env.possible_agents[0])

        return search_engine, network, config, obs

    return _setup


def test_expand_node_decision(search_setup):
    torch.manual_seed(42)
    np.random.seed(42)
    search_engine, network, config, obs = search_setup()

    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
    out = network.obs_inference(obs_tensor)

    root = DecisionNode(prior=1.0)
    root.expand(
        allowed_actions=list(range(search_engine.num_actions)),
        to_play=0,
        priors=out.policy.probs[0],
        network_policy=out.policy.probs[0],
        network_policy_dist=out.policy,
        network_state=out.network_state,
        reward=0.0,
        value=out.value[0].item(),
    )
    root.visits = 1

    child = DecisionNode(prior=0.1, parent=root)
    action_idx = 3
    root.children[action_idx] = child

    value, to_play = search_engine._expand_node(child, root, action_idx, network)

    assert child.expanded()
    assert abs(child.network_value) <= 10.0  # Real network random values
    assert abs(value) <= 10.0
    assert to_play == 0


def test_select_child(search_setup):
    torch.manual_seed(42)
    np.random.seed(42)
    search_engine, network, config, obs = search_setup()
    min_max_stats = MinMaxStats(config.known_bounds)

    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
    out = network.obs_inference(obs_tensor)

    root = DecisionNode(prior=1.0)
    root.expand(
        allowed_actions=list(range(search_engine.num_actions)),
        to_play=0,
        priors=out.policy.probs[0],
        network_policy=out.policy.probs[0],
        network_policy_dist=out.policy,
        network_state=out.network_state,
        reward=0.0,
        value=out.value[0].item(),
    )
    root.visits = 1

    search_path = [root]
    action_path = []
    pruning_context = {"root": None, "internal": {}}

    node, sp, ap, action = search_engine._select_child(
        root, search_path, action_path, min_max_stats, 0, pruning_context
    )

    assert len(sp) == 2
    assert sp[0] == root
    assert isinstance(sp[1], DecisionNode)
    assert len(ap) == 1
    assert action == ap[0]


def test_backpropagate(search_setup):
    torch.manual_seed(42)
    np.random.seed(42)
    search_engine, network, config, obs = search_setup()
    min_max_stats = MinMaxStats(config.known_bounds)

    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
    out = network.obs_inference(obs_tensor)

    root = DecisionNode(prior=1.0)
    root.to_play = 0
    root.expand(
        allowed_actions=list(range(search_engine.num_actions)),
        to_play=0,
        priors=out.policy.probs[0],
        network_policy=out.policy.probs[0],
        network_policy_dist=out.policy,
        network_state=out.network_state,
        reward=0.0,
        value=out.value[0].item(),
    )
    root.visits = 1

    child = DecisionNode(prior=0.1, parent=root)
    child.to_play = 1
    action_idx = 1
    root.children[action_idx] = child

    search_path = [root, child]
    action_path = [action_idx]

    search_engine._backpropagate(search_path, action_path, 1.0, 1, min_max_stats)

    assert child.visits == 1
    assert root.visits == 2
    assert root.child_visits[action_idx] == 1
    assert root.child_values[action_idx] != 0.0


def test_stochastic_expansion(search_setup):
    torch.manual_seed(42)
    np.random.seed(42)
    search_engine, network, config, obs = search_setup(stochastic=True)

    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
    out = network.obs_inference(obs_tensor)

    root = DecisionNode(prior=1.0)
    root.expand(
        allowed_actions=list(range(search_engine.num_actions)),
        to_play=0,
        priors=out.policy.probs[0],
        network_policy=out.policy.probs[0],
        network_policy_dist=out.policy,
        network_state=out.network_state,
        reward=0.0,
        value=out.value[0].item(),
    )
    root.visits = 1

    chance_node = ChanceNode(prior=0.1, parent=root)
    action_idx = 2
    root.children[action_idx] = chance_node

    value, to_play = search_engine._expand_node(chance_node, root, action_idx, network)

    assert chance_node.expanded()
    assert chance_node.is_chance
    assert chance_node.child_priors.shape[0] == 2
    assert abs(value - 0.5) < 1.0  # Real network outputs are random

    decision_node = DecisionNode(prior=0.5, parent=chance_node)
    code_idx = 1
    chance_node.children[code_idx] = decision_node

    value_d, to_play_d = search_engine._expand_node(
        decision_node, chance_node, code_idx, network
    )

    assert decision_node.expanded()
    assert decision_node.is_decision
    assert abs(value_d - 0.5) < 1.0


def test_batched_simulations_coverage(search_setup):
    """Explicitly tests lines 735-865 of modular_search.py via search_batch_size > 0."""
    torch.manual_seed(42)
    np.random.seed(42)
    search_engine, network, config, obs = search_setup()

    # Force batched simulation loop
    config.search_batch_size = 2
    config.num_simulations = 4

    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
    info = {"legal_moves": [list(range(search_engine.num_actions))], "player": 0}

    # This calls search.run -> _run_batched_simulations
    val, exploratory, target, best_action, meta = search_engine.run(
        obs_tensor[0], info, network
    )

    assert target.shape == (search_engine.num_actions,)
    assert target.sum() == pytest.approx(1.0)
    assert best_action in range(search_engine.num_actions)


def test_batched_vectorized_simulations_coverage(search_setup):
    """Tests the vectorized batched simulation loop in modular_search.py."""
    torch.manual_seed(42)
    np.random.seed(42)
    search_engine, network, config, obs = search_setup()

    # Force batched simulation loop in vectorized mode
    config.search_batch_size = 2
    config.num_simulations = 4

    B = 2
    obs_batch = torch.from_numpy(obs).unsqueeze(0).repeat(B, 1, 1, 1).float()
    info_batch = [
        {"legal_moves": list(range(search_engine.num_actions)), "player": 0}
        for _ in range(B)
    ]

    # This calls search.run_vectorized -> _run_batched_vectorized_simulations
    (
        root_values,
        exploratory_policies,
        target_policies,
        best_actions,
        search_metadata,
    ) = search_engine.run_vectorized(obs_batch, info_batch, network)

    assert len(target_policies) == B
    for i in range(B):
        assert target_policies[i].shape == (search_engine.num_actions,)
        assert target_policies[i].sum() == pytest.approx(1.0)
