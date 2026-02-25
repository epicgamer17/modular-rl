from configs.agents.muzero import MuZeroConfig
from configs.games.game import GameConfig


def test_puffer_config_defaults():
    # Mock game config
    class MockGame:
        def __init__(self):
            self.num_actions = 9
            self.num_players = 2
            self.min_score = -1
            self.max_score = 1
            self.make_env = lambda: None

    game_cfg = MockGame()

    # Minimal config dict
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

    config = MuZeroConfig(config_dict, game_cfg)

    print(f"Executor Type: {config.execution.executor_type}")
    print(f"Num Workers: {config.execution.num_workers}")
    print(f"Num Envs Per Worker: {config.execution.num_envs_per_worker}")
    print(f"Num Puffer Threads: {config.execution.num_puffer_threads}")

    assert config.execution.executor_type == "puffer"
    assert config.execution.num_workers == 4
    assert config.execution.num_envs_per_worker == 32
    assert config.execution.num_puffer_threads == 2

    # Test overriding
    config_dict_override = config_dict.copy()
    config_dict_override["num_workers"] = 8
    config_dict_override["executor_type"] = "torch_mp"

    config_override = MuZeroConfig(config_dict_override, game_cfg)
    assert config_override.execution.num_workers == 8
    assert config_override.execution.executor_type == "torch_mp"

    print("Verification successful!")


if __name__ == "__main__":
    test_puffer_config_defaults()
