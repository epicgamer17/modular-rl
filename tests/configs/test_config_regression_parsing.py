import pytest
pytestmark = pytest.mark.unit

from configs.agents.muzero import MuZeroConfig
from configs.games.tictactoe import TicTacToeConfig


def test_config_parsing():
    user_config = {
        "backbone": {
            "type": "resnet",
            "filters": [24],
            "kernel_sizes": [3],
            "strides": [1],
        },
        "architecture": {
            "value_head": {"widths": [256]},
            "policy_head": {"widths": [256]},
        },
        "search": {"num_simulations": 25, "batch_size": 5, "use_virtual_mean": True},
        "action_selector": {"base": {"type": "argmax", "kwargs": {}}},
    }

    game_config = TicTacToeConfig()
    muzero_cfg = MuZeroConfig(user_config, game_config)

    assert muzero_cfg.num_simulations == 25
    assert muzero_cfg.use_virtual_mean is True
    assert muzero_cfg.value_head is not None
    assert muzero_cfg.policy_head is not None


if __name__ == "__main__":
    test_config_parsing()
