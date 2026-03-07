import pytest
import torch
from configs.agents.ppo import PPOConfig
from configs.agents.muzero import MuZeroConfig
from configs.agents.rainbow_dqn import RainbowConfig
from configs.agents.nfsp import NFSPDQNConfig
from configs.agents.supervised import SupervisedConfig

pytestmark = pytest.mark.unit


def test_ppo_fixture(ppo_config):
    assert isinstance(ppo_config, PPOConfig)
    assert ppo_config.steps_per_epoch == 2
    # Verify that forbidden fields are NOT in the dict after PPOConfig processing
    # Actually PPOConfig._init_ deletes them or derived them?
    # It derived them: config_dict["minibatch_size"] = config_dict["steps_per_epoch"]
    assert ppo_config.minibatch_size == 2


def test_muzero_fixture(muzero_config):
    assert isinstance(muzero_config, MuZeroConfig)
    assert muzero_config.unroll_steps == 3
    assert muzero_config.num_simulations == 3


def test_rainbow_fixture(rainbow_config):
    assert isinstance(rainbow_config, RainbowConfig)
    assert rainbow_config.minibatch_size == 2
    assert rainbow_config.dueling is True


def test_nfsp_fixture(nfsp_config):
    assert isinstance(nfsp_config, NFSPDQNConfig)
    assert nfsp_config.anticipatory_param == 0.1


def test_supervised_fixture(supervised_config):
    assert isinstance(supervised_config, SupervisedConfig)
    assert supervised_config.learning_rate == 1e-3
