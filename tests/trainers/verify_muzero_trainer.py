"""
Minimal smoke test for MuZeroTrainer using real config files.
"""

import sys
from unittest.mock import MagicMock

# Mock matplotlib before any other imports that might load it
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()

import torch
from agents.trainers.muzero_trainer import MuZeroTrainer
from configs.agents.muzero import MuZeroConfig
from configs.games.cartpole import CartPoleConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel


def build_minimal_config():
    """Build a minimal MuZeroConfig using CartPole game."""
    game_config = CartPoleConfig()

    config_dict = {
        "model_name": "test_muzero_trainer",
        "training_steps": 2,
        "min_replay_buffer_size": 1,
        "minibatch_size": 2,
        "num_minibatches": 1,
        "replay_buffer_size": 50,
        "unroll_steps": 1,
        "n_step": 1,
        "discount_factor": 0.99,
        "multi_process": False,
        "num_workers": 1,
        "world_model_cls": MuzeroWorldModel,
        # Simplified network architecture
        "residual_layers": [],
        "representation_residual_layers": [],
        "dynamics_residual_layers": [],
        "dense_layer_widths": [16],
        "representation_dense_layer_widths": [16],
        "dynamics_dense_layer_widths": [16],
        "reward_dense_layer_widths": [16],
        "to_play_dense_layer_widths": [16],
        "critic_dense_layer_widths": [16],
        "actor_dense_layer_widths": [16],
        "reward_conv_layers": [],
        "to_play_conv_layers": [],
        "critic_conv_layers": [],
        "actor_conv_layers": [],
    }

    return MuZeroConfig(config_dict, game_config)


class MockStats:
    def __init__(self, *args, **kwargs):
        self._is_client = False

    def append(self, *args, **kwargs):
        pass

    def get_last(self, key):
        return 0.0

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

    def add_latent_visualization(self, *args, **kwargs):
        pass

    def add_custom_visualization(self, *args, **kwargs):
        pass


def test_muzero_trainer_init():
    print("Building config...", flush=True)
    config = build_minimal_config()
    print("Creating environment...", flush=True)
    env = config.game.make_env()
    print("Initializing MuZeroTrainer...", flush=True)
    trainer = MuZeroTrainer(config, env, torch.device("cpu"), stats=MockStats())
    assert trainer.learner is not None
    assert trainer.executor is not None
    assert trainer.buffer is not None
    trainer.executor.stop()
    env.close()
    print("MuZeroTrainer init test passed!", flush=True)


if __name__ == "__main__":
    print("Starting verification...", flush=True)
    try:
        test_muzero_trainer_init()
        print("All tests passed!", flush=True)
    except Exception as e:
        print(f"Verification failed: {e}", flush=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)
