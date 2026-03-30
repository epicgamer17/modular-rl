import pytest

from agents.factories.search import SearchBackendFactory
from configs.agents.muzero import MuZeroConfig
from configs.games.tictactoe import TicTacToeConfig
from search.search_py.min_max_stats import MinMaxStats

pytestmark = pytest.mark.integration


def get_base_muzero_config_dict():
    """A complete, minimal valid config for deterministic MuZero search setup."""
    return {
        "minibatch_size": 2,
        "min_replay_buffer_size": 0,
        "num_simulations": 3,
        "discount_factor": 0.99,
        "unroll_steps": 3,
        "lr_init": 0.01,
        "architecture": {"backbone": {"type": "identity"}},
        "action_selector": {"base": {"type": "categorical"}},
        "policy_head": {
            "output_strategy": {"type": "categorical"},
            "neck": {"type": "identity"},
        },
        "value_head": {
            "output_strategy": {"type": "muzero"},
            "neck": {"type": "identity"},
        },
        "reward_head": {
            "output_strategy": {"type": "muzero"},
            "neck": {"type": "identity"},
        },
        "agent_type": "muzero",
        "to_play_head": {"output_strategy": {"type": "categorical"}},
        "executor_type": "local",
        "search_backend": "python",
        "use_dirichlet": False,
    }


def test_muzero_tictactoe_known_bounds_seed_min_max_stats_immediately():
    """
    Tier 2 integration test:
    MuZero Tic-Tac-Toe search should carry the configured game bounds [-1, 1]
    into MinMaxStats so the first normalization is centered at 0.5, not the
    uninitialized fallback of 0.0.
    """
    game_config = TicTacToeConfig()
    config_dict = get_base_muzero_config_dict()
    config_dict["known_bounds"] = [game_config.min_score, game_config.max_score]

    config = MuZeroConfig(config_dict, game_config)
    search = SearchBackendFactory.create(config)

    assert search.config.known_bounds == [-1, 1]

    stats = MinMaxStats(
        known_bounds=search.config.known_bounds,
        epsilon=search.config.min_max_epsilon,
    )

    assert stats.min == -1.0
    assert stats.max == 1.0
    assert stats.normalize(-1.0) == 0.0
    assert stats.normalize(0.0) == pytest.approx(0.5)
    assert stats.normalize(1.0) == 1.0
