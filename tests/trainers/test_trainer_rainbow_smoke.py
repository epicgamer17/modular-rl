"""
Minimal smoke tests for RainbowTrainer after the UniversalLearner refactor.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from agents.learners.batch_iterators import RepeatSampleIterator
from agents.trainers.rainbow_trainer import RainbowTrainer
from configs.agents.rainbow_dqn import RainbowConfig
from configs.games.cartpole import CartPoleConfig


class MockStats:
    def __init__(self, *args, **kwargs):
        self.stats = {}
        self._is_client = False

    def append(self, *args, **kwargs):
        pass

    def _init_key(self, *args, **kwargs):
        pass

    def drain_queue(self, *args, **kwargs):
        pass

    def add_plot_types(self, *args, **kwargs):
        pass

    def set(self, *args, **kwargs):
        pass

    def add_latent_visualization(self, *args, **kwargs):
        pass

    def get_data(self):
        return {}

    def plot_graphs(self, *args, **kwargs):
        pass


def build_minimal_config(make_rainbow_config_dict, atom_size=1):
    game_config = CartPoleConfig()
    config_dict = make_rainbow_config_dict(
        training_steps=2,
        min_replay_buffer_size=1,
        minibatch_size=2,
        num_minibatches=1,
        training_iterations=1,
        replay_buffer_size=50,
        executor_type="local",
        num_workers=1,
        atom_size=atom_size,
        transfer_interval=1,
        replay_interval=1,
        epsilon_schedule={
            "type": "linear",
            "initial": 1.0,
            "final": 0.05,
            "decay_steps": 1000,
        },
        action_selector={
            "base": {"type": "epsilon_greedy", "kwargs": {"epsilon": 0.05}}
        },
    )
    return RainbowConfig(config_dict, game_config)


def _populate_cartpole_like_transitions(buffer, num_actions: int, n: int = 10):
    for _ in range(n):
        obs = np.random.randn(4).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)
        buffer.store(
            observations=obs,
            actions=0,
            rewards=1.0,
            next_observations=next_obs,
            next_legal_moves=list(range(num_actions)),
            terminated=False,
            truncated=False,
            dones=False,
        )


def test_rainbow_trainer_init(make_rainbow_config_dict):
    config = build_minimal_config(make_rainbow_config_dict)
    env = config.game.make_env()
    trainer = RainbowTrainer(
        config=config,
        env=env,
        device=torch.device("cpu"),
        name="test_rainbow_trainer",
        stats=MockStats(),
    )

    assert trainer.learner is not None
    assert trainer.executor is not None
    assert trainer.buffer is not None
    assert trainer.action_selector is not None

    trainer.executor.stop()
    env.close()


def test_rainbow_trainer_epsilon_update(make_rainbow_config_dict):
    config = build_minimal_config(make_rainbow_config_dict)
    env = config.game.make_env()
    trainer = RainbowTrainer(
        config=config,
        env=env,
        device=torch.device("cpu"),
        name="test_rainbow_trainer",
        stats=MockStats(),
    )

    assert trainer.current_epsilon == pytest.approx(1.0)

    trainer.schedules["epsilon"].step()
    assert trainer.current_epsilon < 1.0
    assert trainer.current_epsilon > 0.05

    trainer.executor.stop()
    env.close()


def test_rainbow_trainer_scalar_dqn_learner_step(make_rainbow_config_dict):
    config = build_minimal_config(make_rainbow_config_dict, atom_size=1)
    env = config.game.make_env()
    trainer = RainbowTrainer(
        config=config,
        env=env,
        device=torch.device("cpu"),
        name="test_rainbow_trainer",
        stats=MockStats(),
    )

    _populate_cartpole_like_transitions(trainer.buffer, trainer.num_actions, n=10)
    iterator = RepeatSampleIterator(trainer.buffer, trainer.config.training_iterations, trainer.device)
    loss_stats = list(trainer.learner.step(batch_iterator=iterator))

    assert loss_stats
    assert "loss" in loss_stats[0]

    trainer.executor.stop()
    env.close()


def test_rainbow_trainer_c51_learner_step(make_rainbow_config_dict):
    config = build_minimal_config(make_rainbow_config_dict, atom_size=51)
    env = config.game.make_env()
    trainer = RainbowTrainer(
        config=config,
        env=env,
        device=torch.device("cpu"),
        name="test_rainbow_trainer",
        stats=MockStats(),
    )

    _populate_cartpole_like_transitions(trainer.buffer, trainer.num_actions, n=10)
    iterator = RepeatSampleIterator(trainer.buffer, trainer.config.training_iterations, trainer.device)
    loss_stats = list(trainer.learner.step(batch_iterator=iterator))

    assert loss_stats
    assert "loss" in loss_stats[0]

    trainer.executor.stop()
    env.close()
