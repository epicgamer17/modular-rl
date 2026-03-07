import gymnasium
import pytest
import torch
import torch.nn.functional as F
import numpy as np
import random
from agents.trainers.ppo_trainer import PPOTrainer
from agents.trainers.rainbow_trainer import RainbowTrainer
from agents.trainers.muzero_trainer import MuZeroTrainer
from agents.trainers.nfsp_trainer import NFSPTrainer
from configs.agents.ppo import PPOConfig
from configs.agents.rainbow_dqn import RainbowConfig
from configs.agents.muzero import MuZeroConfig
from configs.agents.nfsp import NFSPDQNConfig

pytestmark = pytest.mark.integration


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_ppo_trainer_micro_step(make_cartpole_config):
    set_seeds()
    device = torch.device("cpu")
    game_config = make_cartpole_config()
    game_config.training_steps = 10

    # Fix: Replay buffer size must match steps_per_epoch for PPO if not specified
    # Actually PPOConfig auto-sets it.
    config_dict = {
        "training_steps": 10,
        "steps_per_epoch": 2,
        "train_policy_iterations": 1,
        "train_value_iterations": 1,
        "action_selector": {"base": {"type": "categorical", "kwargs": {}}},
        "actor_config": {"clipnorm": 0, "learning_rate": 0.0003},
        "critic_config": {"clipnorm": 0, "learning_rate": 0.0003},
    }

    config = PPOConfig(config_dict, game_config)
    env = game_config.make_env()

    trainer = PPOTrainer(config=config, env=env, device=device)
    trainer.setup()

    initial_step = trainer.training_step
    trainer.train_step()

    assert trainer.training_step == initial_step + 1
    # Check that some stats were recorded
    assert any(
        key in trainer.stats.stats for key in ["score", "learner_fps", "policy_loss"]
    )


def test_rainbow_trainer_micro_step(make_cartpole_config):
    set_seeds()
    device = torch.device("cpu")
    game_config = make_cartpole_config()
    game_config.training_steps = 10

    config_dict = {
        "training_steps": 10,
        "min_replay_buffer_size": 1,
        "replay_buffer_size": 10,
        "minibatch_size": 2,
        "num_minibatches": 1,
        "multi_process": False,
        "num_workers": 1,
        "action_selector": {"base": {"type": "epsilon_greedy", "kwargs": {}}},
        "per_alpha": 0.6,
        "per_beta_schedule": {"type": "constant", "initial": 0.4},
        "per_epsilon": 1e-6,
        "per_use_batch_weights": True,
        "per_use_initial_max_priority": True,
        "n_step": 1,
        "replay_interval": 1,
        "clipnorm": 0,
    }

    config = RainbowConfig(config_dict, game_config)
    env = game_config.make_env()

    trainer = RainbowTrainer(config=config, env=env, device=device)
    trainer.setup()

    # For Rainbow, we need to collect at least one sample to increment step
    initial_step = trainer.training_step
    trainer.train_step()

    print(f"DEBUG: Rainbow buffer size: {trainer.buffer.size}")
    assert trainer.training_step == initial_step + 1
    assert trainer.buffer.size > 0


def test_muzero_trainer_micro_step(make_cartpole_config):
    set_seeds()
    device = torch.device("cpu")
    game_config = make_cartpole_config()
    game_config.training_steps = 10

    config_dict = {
        "training_steps": 10,
        "min_replay_buffer_size": 1,
        "replay_buffer_size": 10,
        "minibatch_size": 2,
        "num_minibatches": 1,
        "num_simulations": 2,
        "multi_process": False,
        "num_workers": 1,
        "unroll_steps": 1,
        "td_steps": 1,
        "backbone": {"type": "dense", "hidden_sizes": [8]},
        "action_selector": {"base": {"type": "mcts", "kwargs": {"num_simulations": 2}}},
        "replay_interval": 1,
        "clipnorm": 0,
    }

    config = MuZeroConfig(config_dict, game_config)
    env = game_config.make_env()

    trainer = MuZeroTrainer(config=config, env=env, device=device)
    trainer.setup()

    initial_step = trainer.training_step
    trainer.train_step()

    assert trainer.training_step == initial_step + 1
    assert trainer.buffer.size > 0


def test_nfsp_trainer_micro_step(make_cartpole_config):
    set_seeds()
    device = torch.device("cpu")
    game_config = make_cartpole_config()
    game_config.training_steps = 10

    config_dict = {
        "training_steps": 10,
        "replay_interval": 1,
        "num_minibatches": 1,
        "multi_process": False,
        "num_workers": 1,
        "min_replay_buffer_size": 1,
        "replay_buffer_size": 10,
        "action_selector": {"base": {"type": "nfsp", "kwargs": {"eta": 0.1}}},
        "sl_loss_function": F.mse_loss,
        "training_steps": 10,
        "clipnorm": 0,
        "sl_clipnorm": 0,
    }

    config = NFSPDQNConfig(config_dict, game_config)
    env = game_config.make_env()

    trainer = NFSPTrainer(config=config, env=env, device=device)
    trainer.setup()

    initial_step = trainer.training_step
    trainer.train_step()

    assert trainer.training_step == initial_step + 1
