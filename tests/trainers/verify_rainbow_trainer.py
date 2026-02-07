"""
Minimal smoke test for RainbowTrainer using real config files.
"""

import sys
from unittest.mock import MagicMock

# Mock matplotlib before any other imports that might load it
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()

import torch
from trainers.rainbow_trainer import RainbowTrainer
from agent_configs.dqn.rainbow_config import RainbowConfig
from game_configs.cartpole_config import CartPoleConfig


def build_minimal_config():
    """Build a minimal RainbowConfig using CartPole game."""
    game_config = CartPoleConfig()

    config_dict = {
        "model_name": "test_rainbow_trainer",
        "training_steps": 2,
        "min_replay_buffer_size": 1,
        "minibatch_size": 2,
        "num_minibatches": 1,
        "training_iterations": 1,
        "replay_buffer_size": 50,
        "multi_process": False,
        "num_workers": 1,
        # Simplified network architecture
        "residual_layers": [],
        "conv_layers": [],
        "dense_layer_widths": [16],
        "value_hidden_layer_widths": [16],
        "advantage_hidden_layer_widths": [16],
        # DQN-specific
        "atom_size": 1,  # Non-distributional for simplicity
        "transfer_interval": 1,
        "replay_interval": 1,
        # Epsilon greedy
        "eg_epsilon": 1.0,
        "eg_epsilon_final": 0.05,
        "eg_epsilon_final_step": 1000,
        "eg_epsilon_decay_type": "linear",
    }

    return RainbowConfig(config_dict, game_config)


class MockStats:
    """Mock StatTracker for testing."""

    def __init__(self, *args, **kwargs):
        self.stats = {}
        self._is_client = False

    def append(self, *args, **kwargs):
        pass

    def _init_key(self, *args, **kwargs):
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

    def get_data(self):
        return {}

    def plot_graphs(self, *args, **kwargs):
        pass


def test_rainbow_trainer_init():
    """Test that RainbowTrainer initializes correctly."""
    print("Building config...", flush=True)
    config = build_minimal_config()

    print("Creating environment...", flush=True)
    env = config.game.make_env()

    print("Initializing RainbowTrainer...", flush=True)
    trainer = RainbowTrainer(config, env, torch.device("cpu"), stats=MockStats())

    assert trainer.learner is not None, "Learner should be initialized"
    assert trainer.executor is not None, "Executor should be initialized"
    assert trainer.buffer is not None, "Buffer should be initialized"
    assert trainer.policy is not None, "Policy should be initialized"
    assert trainer.action_selector is not None, "ActionSelector should be initialized"

    trainer.executor.stop()
    env.close()
    print("RainbowTrainer init test passed!", flush=True)


def test_rainbow_trainer_epsilon_update():
    """Test that epsilon is correctly updated during training."""
    print("Testing epsilon schedule...", flush=True)
    config = build_minimal_config()
    env = config.game.make_env()
    trainer = RainbowTrainer(config, env, torch.device("cpu"), stats=MockStats())

    initial_epsilon = trainer.current_epsilon
    assert (
        initial_epsilon == 1.0
    ), f"Initial epsilon should be 1.0, got {initial_epsilon}"

    # Simulate training step
    trainer.training_step = 500
    trainer._update_epsilon()

    assert trainer.current_epsilon < initial_epsilon, "Epsilon should decrease"
    assert trainer.current_epsilon > 0.05, "Epsilon should not be at minimum yet"

    trainer.executor.stop()
    env.close()
    print("Epsilon schedule test passed!", flush=True)


if __name__ == "__main__":
    print("Starting RainbowTrainer verification...", flush=True)
    try:
        test_rainbow_trainer_init()
        test_rainbow_trainer_epsilon_update()
        print("All tests passed!", flush=True)
    except Exception as e:
        print(f"Verification failed: {e}", flush=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)
