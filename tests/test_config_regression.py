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
    }

    game_config = TicTacToeConfig()
    muzero_cfg = MuZeroConfig(user_config, game_config)

    print(f"Value Head Neck: {muzero_cfg.value_head.neck}")
    print(f"Policy Head Neck: {muzero_cfg.policy_head.neck}")

    # Check if widths [256] made it through
    # Heads currently use 'neck' which defaults to resnet with [16] filters if not specified?
    # Or does it use dense if widths is specified but type is missing?


if __name__ == "__main__":
    test_config_parsing()
