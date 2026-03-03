import os
import sys
import torch
import numpy as np
from agents.trainers.muzero_trainer import MuZeroTrainer
from configs.agents.muzero import MuZeroConfig
from configs.games.tictactoe import TicTacToeConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from agents.tictactoe_expert import TicTacToeBestAgent


def build_config():
    game_config = TicTacToeConfig()

    config_dict = {
        "training_steps": 2000,
        "min_replay_buffer_size": 200,
        "minibatch_size": 128,
        "num_minibatches": 4,
        "replay_buffer_size": 10000,
        "unroll_steps": 5,
        "n_step": 3,
        "discount_factor": 0.99,
        "multi_process": True,
        "num_workers": 4,
        "search_batch_size": 5,
        "num_simulations": 100,
        "test_interval": 500,
        "test_trials": 20,
        "world_model_cls": MuzeroWorldModel,
        "action_selector": {
            "base": {"type": "categorical", "kwargs": {"exploration": True}}
        },
        "backbone": {"type": "dense", "widths": [64, 64]},
        "reward_head": {
            "neck": {"type": "dense", "widths": [64]},
            "output_strategy": {"type": "scalar"},
        },
        "value_head": {
            "neck": {"type": "dense", "widths": [64]},
            "output_strategy": {"type": "scalar"},
        },
        "policy_head": {"neck": {"type": "dense", "widths": [64]}},
        "to_play_head": {"neck": {"type": "dense", "widths": [64]}},
    }

    return MuZeroConfig(config_dict, game_config)


class ConsoleStats:
    def __init__(self):
        self.stats = {}
        self._is_client = False

    def _init_key(self, key, subkeys=None):
        if subkeys:
            self.stats[key] = {sk: [] for sk in subkeys}
        else:
            self.stats[key] = []

    def append(self, key, value, step=None):
        if key not in self.stats:
            self.stats[key] = []

        if isinstance(value, dict):
            if key not in self.stats:
                self.stats[key] = {sk: [] for sk in value.keys()}
            for sk, sv in value.items():
                self.stats[key][sk].append(sv)
        else:
            self.stats[key].append(value)

        if "test_score" in key and isinstance(value, dict):
            print(f"Step {step} - {key}: {value}")

    def increment_steps(self, *args, **kwargs):
        pass

    def get_num_steps(self):
        return 0

    def get_time_elapsed(self):
        return 0.0

    def set_time_elapsed(self, *args, **kwargs):
        pass

    def drain_queue(self, *args, **kwargs):
        pass

    def add_plot_types(self, *args, **kwargs):
        pass


def main():
    print("Building config...")
    config = build_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # We want to test vs expert
    expert = TicTacToeBestAgent()

    # Standard MuZeroTrainer expects list of test agents
    # But we can also just let it run self-play tests

    env = config.game.make_env()
    trainer = MuZeroTrainer(config, env, device, name="repro_p2", stats=ConsoleStats())

    print("Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        trainer.executor.stop()
        env.close()


if __name__ == "__main__":
    main()
