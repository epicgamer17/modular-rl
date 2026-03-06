import copy
import pytest

from configs.agents.rainbow_dqn import RainbowConfig
from configs.games.cartpole import CartPoleConfig


@pytest.fixture(scope="session")
def cartpole_game_config():
    """Real, lightweight single-agent game configuration used across tests."""
    return CartPoleConfig()


@pytest.fixture(scope="session")
def rainbow_cartpole_replay_config(cartpole_game_config):
    """
    Real RainbowConfig fixture for replay-buffer tests.
    This replaces ad-hoc synthetic config objects.
    """
    config_dict = {
        "action_selector": {"base": {"type": "epsilon_greedy", "kwargs": {}}},
        "executor_type": "local",
        "n_step": 3,
        "discount_factor": 0.9,
        "per_alpha": 0.6,
        "per_beta_schedule": {"type": "constant", "initial": 0.4},
        "per_epsilon": 1e-6,
        "per_use_batch_weights": True,
        "per_use_initial_max_priority": True,
        "minibatch_size": 32,
        "replay_buffer_size": 100,
        "min_replay_buffer_size": 1,
    }
    return RainbowConfig(config_dict, cartpole_game_config)


@pytest.fixture
def make_cartpole_config(cartpole_game_config):
    def _builder(**overrides):
        config = copy.deepcopy(cartpole_game_config)
        for attr, value in overrides.items():
            setattr(config, attr, value)
        return config

    return _builder
