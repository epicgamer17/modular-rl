import copy
import pytest
import torch
import torch.nn.functional as F

from configs.agents.ppo import PPOConfig
from configs.agents.muzero import MuZeroConfig
from configs.agents.rainbow_dqn import RainbowConfig
from configs.agents.nfsp import NFSPDQNConfig
from configs.agents.supervised import SupervisedConfig
from configs.games.cartpole import CartPoleConfig


# --- BASE DICTIONARIES ---


@pytest.fixture(scope="session")
def base_ppo_config_dict():
    """A complete, minimal valid config for PPO."""
    return {
        "steps_per_epoch": 2,
        "clip_param": 0.2,
        "entropy_coefficient": 0.01,
        "critic_coefficient": 0.5,
        "discount_factor": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 1e-3,
        "adam_epsilon": 1e-8,
        "actor_config": {
            "optimizer": torch.optim.Adam,
            "learning_rate": 1e-3,
            "adam_epsilon": 1e-8,
            "clipnorm": 0,
        },
        "critic_config": {
            "optimizer": torch.optim.Adam,
            "learning_rate": 1e-3,
            "adam_epsilon": 1e-8,
            "clipnorm": 0,
        },
        "action_selector": {
            "base": {"type": "categorical"},
            "decorators": [{"type": "ppo_injector"}],
        },
        "policy_head": {
            "output_strategy": {"type": "categorical"},
            "neck": {"type": "identity"},
        },
        "value_head": {
            "output_strategy": {"type": "scalar"},
            "neck": {"type": "identity"},
        },
        "architecture": {"backbone": {"type": "identity"}},
    }


@pytest.fixture(scope="session")
def base_muzero_config_dict():
    """A complete, minimal valid config for MuZero."""
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
    }


@pytest.fixture(scope="session")
def base_rainbow_config_dict():
    """A complete, minimal valid config for Rainbow DQN."""
    return {
        "batch_size": 2,
        "minibatch_size": 2,
        "min_replay_buffer_size": 0,
        "replay_buffer_size": 100,
        "learning_rate": 1e-3,
        "adam_epsilon": 1e-8,
        "discount_factor": 0.99,
        "n_step": 3,
        "action_selector": {"base": {"type": "epsilon_greedy", "kwargs": {}}},
        "architecture": {"backbone": {"type": "identity"}},
        "dueling": True,
        "noisy_sigma": 0.5,
    }


@pytest.fixture(scope="session")
def base_nfsp_config_dict(base_rainbow_config_dict):
    """A complete, minimal valid config for NFSP."""
    d = copy.deepcopy(base_rainbow_config_dict)
    d.update(
        {
            "anticipatory_param": 0.1,
            "sl_learning_rate": 1e-3,
            "sl_loss_function": F.cross_entropy,
            "training_steps": 100,
        }
    )
    return d


@pytest.fixture(scope="session")
def base_supervised_config_dict():
    """A complete, minimal valid config for Supervised learning."""
    return {
        "sl_learning_rate": 1e-3,
        "sl_loss_function": F.cross_entropy,
        "training_steps": 100,
        "architecture": {"backbone": {"type": "identity"}},
    }


# --- FACTORY FIXTURES ---


@pytest.fixture
def make_ppo_config_dict(base_ppo_config_dict):
    """Factory to generate a safe, mutable PPO config dict with overrides."""

    def _builder(**overrides):
        cfg = copy.deepcopy(base_ppo_config_dict)
        cfg.update(overrides)
        return cfg

    return _builder


@pytest.fixture
def make_muzero_config_dict(base_muzero_config_dict):
    """Factory to generate a safe, mutable MuZero config dict with overrides."""

    def _builder(**overrides):
        cfg = copy.deepcopy(base_muzero_config_dict)
        cfg.update(overrides)
        return cfg

    return _builder


@pytest.fixture
def make_rainbow_config_dict(base_rainbow_config_dict):
    """Factory to generate a safe, mutable Rainbow config dict with overrides."""

    def _builder(**overrides):
        cfg = copy.deepcopy(base_rainbow_config_dict)
        cfg.update(overrides)
        return cfg

    return _builder


@pytest.fixture
def make_nfsp_config_dict(base_nfsp_config_dict):
    """Factory to generate a safe, mutable NFSP config dict with overrides."""

    def _builder(**overrides):
        cfg = copy.deepcopy(base_nfsp_config_dict)
        cfg.update(overrides)
        return cfg

    return _builder


@pytest.fixture
def make_supervised_config_dict(base_supervised_config_dict):
    """Factory to generate a safe, mutable Supervised config dict with overrides."""

    def _builder(**overrides):
        cfg = copy.deepcopy(base_supervised_config_dict)
        cfg.update(overrides)
        return cfg

    return _builder


# --- CONFIG OBJECT FIXTURES ---


@pytest.fixture(scope="session")
def cartpole_game_config():
    """Real, lightweight single-agent game configuration used across tests."""
    return CartPoleConfig()


@pytest.fixture
def ppo_config(make_ppo_config_dict, cartpole_game_config):
    """Returns a ready-to-use PPOConfig object for CartPole."""
    return PPOConfig(make_ppo_config_dict(), cartpole_game_config)


@pytest.fixture
def muzero_config(make_muzero_config_dict, cartpole_game_config):
    """Returns a ready-to-use MuZeroConfig object for CartPole."""
    return MuZeroConfig(make_muzero_config_dict(), cartpole_game_config)


@pytest.fixture
def rainbow_config(make_rainbow_config_dict, cartpole_game_config):
    """Returns a ready-to-use RainbowConfig object for CartPole."""
    return RainbowConfig(make_rainbow_config_dict(), cartpole_game_config)


@pytest.fixture(scope="session")
def tictactoe_game_config():
    """Real, lightweight multi-agent game configuration."""
    from configs.games.tictactoe import TicTacToeConfig

    return TicTacToeConfig()


@pytest.fixture
def nfsp_config(make_nfsp_config_dict, cartpole_game_config):
    """Returns a ready-to-use NFSPDQNConfig object for CartPole."""
    return NFSPDQNConfig(make_nfsp_config_dict(), cartpole_game_config)


@pytest.fixture
def supervised_config(make_supervised_config_dict):
    """Returns a ready-to-use SupervisedConfig object."""
    return SupervisedConfig(make_supervised_config_dict())


@pytest.fixture(scope="session")
def rainbow_cartpole_replay_config(cartpole_game_config, base_rainbow_config_dict):
    """
    Real RainbowConfig fixture for replay-buffer tests.
    Maintain backward compatibility while using base dict.
    """
    return RainbowConfig(base_rainbow_config_dict, cartpole_game_config)


@pytest.fixture
def make_cartpole_config(cartpole_game_config):
    def _builder(**overrides):
        config = copy.deepcopy(cartpole_game_config)
        for attr, value in overrides.items():
            setattr(config, attr, value)
        return config

    return _builder
