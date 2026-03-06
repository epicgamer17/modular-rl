"""
Minimal smoke test for MuZeroTrainer using real config files.
"""
import pytest
pytestmark = [pytest.mark.integration, pytest.mark.slow]


import sys
import matplotlib

matplotlib.use("Agg")

import torch
import torch.multiprocessing as mp

try:
    mp.set_sharing_strategy("file_system")
except Exception:
    pass

from agents.trainers.muzero_trainer import MuZeroTrainer
from configs.agents.muzero import MuZeroConfig
from configs.games.cartpole import CartPoleConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel


def build_minimal_config():
    """Build a minimal MuZeroConfig using CartPole game."""
    game_config = CartPoleConfig()

    config_dict = {
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
        "action_selector": {
            "base": {"type": "categorical", "kwargs": {"exploration": True}}
        },
        "backbone": {"type": "dense", "widths": [16]},
        "reward_head": {
            "neck": {"type": "dense", "widths": [16]},
            "output_strategy": {"type": "scalar"},
        },
        "value_head": {
            "neck": {"type": "dense", "widths": [16]},
            "output_strategy": {"type": "scalar"},
        },
        "policy_head": {"neck": {"type": "dense", "widths": [16]}},
        "to_play_head": {"neck": {"type": "dense", "widths": [16]}},
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
    trainer = MuZeroTrainer(
        config, env, torch.device("cpu"), name="test_muzero_trainer", stats=MockStats()
    )
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
