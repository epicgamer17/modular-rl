import pytest
pytestmark = pytest.mark.unit

from configs.agents.muzero import MuZeroConfig


def test_executor_config_defaults(cartpole_game_config):
    config_dict = {
        "action_selector": {
            "type": "categorical",
            "base": {"type": "categorical_selector"},
        },
        "reward_head": {},
        "value_head": {},
        "policy_head": {},
        "representation_backbone": {"type": "dense"},
        "dynamics_backbone": {"type": "dense"},
        "prediction_backbone": {"type": "dense"},
    }

    config = MuZeroConfig(config_dict, cartpole_game_config)

    assert config.executor_type == "torch_mp"
    assert config.num_workers == 4
    assert config.num_envs_per_worker == 1
    assert config.num_puffer_threads == 2

    # Test overriding
    config_dict_override = config_dict.copy()
    config_dict_override["num_workers"] = 8
    config_dict_override["executor_type"] = "torch_mp"

    config_override = MuZeroConfig(config_dict_override, cartpole_game_config)
    assert config_override.num_workers == 8
    assert config_override.executor_type == "torch_mp"


if __name__ == "__main__":
    test_executor_config_defaults()
