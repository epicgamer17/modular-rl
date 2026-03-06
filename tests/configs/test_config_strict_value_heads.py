import pytest

pytestmark = pytest.mark.unit

import copy

from configs.agents.muzero import MuZeroConfig
from configs.agents.ppo import PPOConfig
from configs.modules.output_strategies import MuZeroSupportConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel


def test_ppo_strict_value_head_requires_explicit_strategy(cartpole_game_config):
    base_dict = {
        "atom_size": 51,
        "support_range": 10,
        "value_head": {"neck": {"type": "identity"}},
        "policy_head": {"neck": {"type": "identity"}},
        "actor_config": {"clipnorm": 0.0},
        "critic_config": {"clipnorm": 0.0},
        "steps_per_epoch": 100,
        "train_policy_iterations": 1,
        "train_value_iterations": 1,
        "action_selector": {"base": {"type": "argmax", "kwargs": {}}},
    }

    with pytest.raises(ValueError, match="requires an explicit output_strategy"):
        PPOConfig(copy.deepcopy(base_dict), cartpole_game_config)



def test_ppo_strict_value_head_overrides_strategy_num_classes(cartpole_game_config):
    cfg_dict = {
        "atom_size": 51,
        "support_range": 10,
        "value_head": {
            "neck": {"type": "identity"},
            "output_strategy": {"type": "categorical", "num_classes": 999},
        },
        "policy_head": {"neck": {"type": "identity"}},
        "actor_config": {"clipnorm": 0.0},
        "critic_config": {"clipnorm": 0.0},
        "steps_per_epoch": 100,
        "train_policy_iterations": 1,
        "train_value_iterations": 1,
        "action_selector": {"base": {"type": "argmax", "kwargs": {}}},
    }

    config = PPOConfig(cfg_dict, cartpole_game_config)

    assert config.value_head.output_strategy.num_classes == 51



def test_muzero_strict_value_head_defaults_strategy(cartpole_game_config):
    base_dict = {
        "world_model_cls": MuzeroWorldModel,
        "atom_size": 51,
        "support_range": 10,
        "value_head": {"neck": {"type": "identity"}},
        "policy_head": {"neck": {"type": "identity"}},
        "representation_backbone": {"type": "identity"},
        "dynamics_backbone": {"type": "identity"},
        "prediction_backbone": {"type": "identity"},
        "afterstate_dynamics_backbone": {"type": "identity"},
        "chance_encoder_backbone": {"type": "identity"},
        "action_selector": {"base": {"type": "argmax", "kwargs": {}}},
    }

    config = MuZeroConfig(copy.deepcopy(base_dict), cartpole_game_config)
    assert isinstance(config.value_head.output_strategy, MuZeroSupportConfig)
    assert config.value_head.output_strategy.support_range == 10



def test_muzero_strict_value_head_overrides_strategy_num_classes(cartpole_game_config):
    cfg_dict = {
        "world_model_cls": MuzeroWorldModel,
        "atom_size": 51,
        "support_range": 10,
        "value_head": {
            "neck": {"type": "identity"},
            "output_strategy": {"type": "categorical", "num_classes": 999},
        },
        "policy_head": {"neck": {"type": "identity"}},
        "representation_backbone": {"type": "identity"},
        "dynamics_backbone": {"type": "identity"},
        "prediction_backbone": {"type": "identity"},
        "afterstate_dynamics_backbone": {"type": "identity"},
        "chance_encoder_backbone": {"type": "identity"},
        "action_selector": {"base": {"type": "argmax", "kwargs": {}}},
    }

    config = MuZeroConfig(cfg_dict, cartpole_game_config)

    assert config.value_head.output_strategy.num_classes == 51
