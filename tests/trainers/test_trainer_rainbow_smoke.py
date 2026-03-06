"""
Minimal smoke test for RainbowTrainer using real config files.
"""
import pytest
pytestmark = [pytest.mark.integration, pytest.mark.slow]


import sys
import matplotlib

matplotlib.use("Agg")

import torch
from agents.trainers.rainbow_trainer import RainbowTrainer
from configs.agents.rainbow_dqn import RainbowConfig
from configs.games.cartpole import CartPoleConfig


def build_minimal_config(atom_size=1):
    """Build a minimal RainbowConfig using CartPole game."""
    game_config = CartPoleConfig()

    config_dict = {
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
        "atom_size": atom_size,
        "v_min": -10,
        "v_max": 10,
        "transfer_interval": 1,
        "replay_interval": 1,
        # Epsilon greedy schedule
        "epsilon_schedule": {
            "type": "linear",
            "initial": 1.0,
            "final": 0.05,
            "decay_steps": 1000,
        },
        "action_selector": {
            "base": {"type": "epsilon_greedy", "kwargs": {"epsilon": 0.05}}
        },
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
    trainer = RainbowTrainer(config, env, torch.device("cpu"), name="test_rainbow_trainer", stats=MockStats())

    assert trainer.learner is not None, "Learner should be initialized"
    assert trainer.executor is not None, "Executor should be initialized"
    assert trainer.buffer is not None, "Buffer should be initialized"
    assert trainer.action_selector is not None, "ActionSelector should be initialized"

    trainer.executor.stop()
    env.close()
    print("RainbowTrainer init test passed!", flush=True)


def test_rainbow_trainer_epsilon_update():
    """Test that epsilon is correctly updated during training."""
    print("Testing epsilon schedule...", flush=True)
    config = build_minimal_config()
    env = config.game.make_env()
    trainer = RainbowTrainer(config, env, torch.device("cpu"), name="test_rainbow_trainer", stats=MockStats())

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


def test_rainbow_c51_training():
    """Test that RainbowTrainer can run training steps with C51."""
    print("Testing C51 training loop...", flush=True)
    config = build_minimal_config(atom_size=51)
    env = config.game.make_env()
    trainer = RainbowTrainer(config, env, torch.device("cpu"), name="test_rainbow_trainer", stats=MockStats())

    # Mock some data collection to avoid environment issues in smoke test
    # but we want to test learner.step which calls the loss pipeline
    from replay_buffers.transition import TransitionBatch, Transition
    import numpy as np

    obs = np.random.randn(4)  # CartPole
    next_obs = np.random.randn(4)
    batch = TransitionBatch(
        [
            Transition(
                observation=obs,
                action=0,
                reward=1.0,
                next_observation=next_obs,
                done=False,
                terminated=False,
                truncated=False,
                info={},
            )
            for _ in range(10)
        ]
    )
    trainer._store_transitions(batch)

    assert trainer.buffer.size >= config.min_replay_buffer_size

    print("Running trainer.learner.step()...", flush=True)
    # This will trigger the C51Loss and the predict() call that was failing
    loss_stats = trainer.learner.step(trainer.stats)
    assert loss_stats is not None
    assert "loss" in loss_stats

    trainer.executor.stop()
    env.close()
    print("C51 training test passed!", flush=True)


def test_rainbow_trainer_test_loop():
    """Test that RainbowTrainer.test() runs without error."""
    print("Testing evaluation loop (test())...", flush=True)
    config = build_minimal_config()
    env = config.game.make_env()
    trainer = RainbowTrainer(config, env, torch.device("cpu"), name="test_rainbow_trainer", stats=MockStats())

    # Run a short test
    scores = trainer.test(num_trials=1)

    assert scores is not None, "Test results should not be None"
    assert "score" in scores, "Test results should contain 'score'"

    trainer.executor.stop()
    env.close()
    print("Evaluation loop test passed!", flush=True)


if __name__ == "__main__":
    print("Starting RainbowTrainer verification...", flush=True)
    try:
        test_rainbow_trainer_init()
        test_rainbow_trainer_epsilon_update()
        test_rainbow_c51_training()
        test_rainbow_trainer_test_loop()
        print("All tests passed!", flush=True)
    except Exception as e:
        print(f"Verification failed: {e}", flush=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)
