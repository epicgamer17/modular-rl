"""
Smoke tests for ImitationTrainer to verify initialization and basic training loop.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

from agents.trainers.imitation_trainer import ImitationTrainer
from configs.agents.supervised import SupervisedConfig
import torch.nn.functional as F
from replay_buffers.sequence import Sequence


def make_cartpole():
    """Factory function for CartPole environment."""
    return gym.make("CartPole-v1")


class MinimalGameConfig:
    """Minimal game config for testing."""

    def __init__(self):
        self.num_players = 1
        self.num_actions = 2
        self.is_discrete = True
        self.min_score = -100.0
        self.max_score = 100.0
        self.is_image = False
        self.make_env = make_cartpole


def create_config():
    """Creates a SupervisedConfig for testing."""
    game = MinimalGameConfig()
    sl_dict = {
        "model_name": "imitation_smoke_test",
        "sl_minibatch_size": 2,
        "sl_replay_buffer_size": 10,
        "sl_min_replay_buffer_size": 2,
        "sl_learning_rate": 0.001,
        "sl_adam_epsilon": 1e-8,
        "sl_weight_decay": 0.0,
        "sl_optimizer": torch.optim.Adam,
        "sl_loss_function": F.mse_loss,
        "sl_training_iterations": 1,
        "sl_clipnorm": 1.0,
        "sl_dense_layer_widths": [16],
        "training_steps": 10,
        "sl_replay_interval": 1,
        "multi_process": False,
        "num_workers": 1,
    }
    config = SupervisedConfig(sl_dict)
    config.game = game
    return config


def test_imitation_trainer_init():
    """Test that ImitationTrainer initializes without errors."""
    config = create_config()
    device = torch.device("cpu")

    trainer = ImitationTrainer(
        config=config,
        env=make_cartpole(),
        device=device,
    )

    assert trainer.model is not None
    assert trainer.learner is not None

    trainer.executor.stop()
    print("✓ test_imitation_trainer_init passed")


def test_imitation_trainer_train():
    """Test training loop for ImitationTrainer."""
    config = create_config()
    device = torch.device("cpu")

    trainer = ImitationTrainer(
        config=config,
        env=make_cartpole(),
        device=device,
    )

    # Manually push a mock sequence into the buffer since it needs data to train
    seq = Sequence(num_players=1)
    obs, info = trainer._env.reset()
    seq.append(obs, info)
    for _ in range(5):
        action = 0
        obs, reward, term, trunc, info = trainer._env.step(action)
        seq.append(obs, info, action=action, reward=reward)

    trainer._store_sequence_transitions(seq)

    try:
        trainer.train()
        print("✓ test_imitation_trainer_train passed")
    except Exception as e:
        print(f"✗ test_imitation_trainer_train failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        trainer.executor.stop()


if __name__ == "__main__":
    print("Running Imitation Trainer verification tests...")
    test_imitation_trainer_init()
    test_imitation_trainer_train()
    print("All tests completed.")
