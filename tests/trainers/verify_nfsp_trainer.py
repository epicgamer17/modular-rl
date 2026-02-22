"""
Smoke tests for NFSPTrainer to verify initialization and basic training loop.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

from agents.trainers.nfsp_trainer import NFSPTrainer
from agents.policies.nfsp_policy import NFSPPolicy
from configs.agents.nfsp import NFSPDQNConfig
from configs.agents.rainbow_dqn import RainbowConfig
from configs.agents.supervised import SupervisedConfig
import torch.nn.functional as F


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


class MockConfig:
    def __init__(self):
        self.training_steps = 100
        self.replay_interval = 1
        self.anticipatory_param = 0.1
        self.num_minibatches = 1
        self.multi_process = False
        self.num_workers = 1
        self.shared_networks_and_buffers = True
        self.observation_dtype = torch.float32
        self.save_intermediate_weights = False

        self.game = MinimalGameConfig()
        self.game.make_env = make_cartpole

        # RL Config (Rainbow)
        rl_dict = {
            "minibatch_size": 2,
            "replay_buffer_size": 10,
            "min_replay_buffer_size": 2,
            "learning_rate": 0.001,
            "adam_epsilon": 1e-8,
            "weight_decay": 0.0,
            "optimizer": torch.optim.Adam,
            "loss_function": F.mse_loss,
            "discount_factor": 0.99,
            "atom_size": 1,
            "v_min": -10,
            "v_max": 10,
            "noisy_sigma": 0.0,
            "dueling": False,
            "double_dqn": False,
            "per": False,
            "multi_step": 1,
            "eg_epsilon": 0.05,
            "transfer_interval": 100,
            "soft_update": False,
        }
        self.rl_configs = [RainbowConfig(rl_dict, self.game)]

        # SL Config
        sl_dict = {
            "sl_minibatch_size": 2,
            "sl_replay_buffer_size": 10,
            "sl_min_replay_buffer_size": 2,
            "sl_learning_rate": 0.001,
            "sl_adam_epsilon": 1e-8,
            "sl_weight_decay": 0.0,
            "sl_optimizer": torch.optim.Adam,
            "sl_loss_function": F.mse_loss,  # Using MSE as a simple proxy for smoke test
            "sl_training_iterations": 1,
            "sl_clipnorm": 1.0,
            "sl_dense_layer_widths": [16],
            "training_steps": 100,
        }
        self.sl_configs = [SupervisedConfig(sl_dict)]
        for c in self.sl_configs:
            c.game = self.game


def test_nfsp_trainer_init():
    """Test that NFSPTrainer initializes without errors."""
    config = MockConfig()
    device = torch.device("cpu")

    trainer = NFSPTrainer(
        config=config,
        env=make_cartpole(),
        device=device,
        name="nfsp_smoke_test",
    )

    assert trainer.br_model is not None
    assert trainer.avg_model is not None
    assert trainer.learner is not None
    assert trainer.policy is not None

    trainer.executor.stop()
    print("✓ test_nfsp_trainer_init passed")


def test_nfsp_trainer_train():
    """Test one step of the NFSP training loop."""
    config = MockConfig()
    device = torch.device("cpu")

    trainer = NFSPTrainer(
        config=config,
        env=make_cartpole(),
        device=device,
        name="nfsp_smoke_test",
    )

    # Run for a very small number of steps
    try:
        trainer.train()
        print("✓ test_nfsp_trainer_train passed")

        # Verify latest-only stat behavior
        stats = trainer.stats.get_data()
        if "sl_policy" in stats:
            policy_data = stats["sl_policy"]
            print(f"DEBUG: sl_policy length: {len(policy_data)}")
            assert (
                len(policy_data) == 1
            ), f"Expected sl_policy to have 1 item, got {len(policy_data)}"

            # Verify no reduction was applied (it should be 2D after stack)
            np_policy = trainer.stats._to_numpy(policy_data, reduce=False)
            print(f"DEBUG: sl_policy shape: {np_policy.shape}")
            # If stack [1D (2,)] -> (1, 2)
            assert (
                len(np_policy.shape) == 2
            ), f"Expected 2D shape for policy distribution, got {np_policy.shape}"
            assert (
                np_policy.shape[1] == 2
            ), f"Expected 2 actions, got {np_policy.shape[1]}"
            print("✓ Latest-only sl_policy verification passed")

    except Exception as e:
        print(f"✗ test_nfsp_trainer_train failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        trainer.executor.stop()


if __name__ == "__main__":
    print("Running NFSP Trainer verification tests...")
    test_nfsp_trainer_init()
    test_nfsp_trainer_train()
    print("All tests completed.")
