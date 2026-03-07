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


def test_ppo_trainer_micro_step(make_ppo_config_dict, cartpole_game_config):
    set_seeds()
    device = torch.device("cpu")
    config_dict = make_ppo_config_dict(
        training_steps=10,
        train_policy_iterations=1,
        train_value_iterations=1,
    )
    config = PPOConfig(config_dict, cartpole_game_config)
    env = cartpole_game_config.make_env()

    trainer = PPOTrainer(config=config, env=env, device=device)
    trainer.setup()

    initial_step = trainer.training_step
    trainer.train_step()

    assert trainer.training_step == initial_step + 1
    assert any(
        key in trainer.stats.stats for key in ["score", "learner_fps", "policy_loss"]
    )


def test_rainbow_trainer_micro_step(make_rainbow_config_dict, cartpole_game_config):
    set_seeds()
    device = torch.device("cpu")
    config_dict = make_rainbow_config_dict(
        training_steps=10,
        min_replay_buffer_size=0,
        replay_buffer_size=10,
        minibatch_size=1,
        multi_process=False,
    )
    config = RainbowConfig(config_dict, cartpole_game_config)
    env = cartpole_game_config.make_env()

    trainer = RainbowTrainer(config=config, env=env, device=device)
    trainer.setup()

    initial_step = trainer.training_step
    trainer.train_step()

    assert trainer.training_step == initial_step + 1
    assert trainer.buffer.size > 0


def test_muzero_trainer_micro_step(make_muzero_config_dict, cartpole_game_config):
    set_seeds()
    device = torch.device("cpu")
    config_dict = make_muzero_config_dict(
        training_steps=10,
        min_replay_buffer_size=0,
        replay_buffer_size=10,
        minibatch_size=1,
        num_simulations=2,
        multi_process=False,
        unroll_steps=1,
    )
    config = MuZeroConfig(config_dict, cartpole_game_config)
    env = cartpole_game_config.make_env()

    trainer = MuZeroTrainer(config=config, env=env, device=device)
    trainer.setup()

    initial_step = trainer.training_step
    trainer.train_step()

    assert trainer.training_step == initial_step + 1
    assert trainer.buffer.size > 0


def test_nfsp_trainer_micro_step(make_nfsp_config_dict, tictactoe_game_config):
    set_seeds()
    device = torch.device("cpu")

    # 1. Use factory from conftest.py
    # NFSP requires both rl_configs and sl_configs in its internal structure
    rl_config_dict = {
        "type": "rainbow_dqn",
        "learning_rate": 1e-3,
        "min_replay_buffer_size": 0,
        "action_selector": {
            "base": {"type": "epsilon_greedy", "kwargs": {"epsilon": 0.1}}
        },
    }
    sl_config_dict = {
        "type": "supervised",
        "learning_rate": 1e-3,
    }

    config_dict = make_nfsp_config_dict(
        rl_configs=[rl_config_dict],
        sl_configs=[sl_config_dict],
        replay_interval=1,
        num_minibatches=1,
        training_steps=10,
        multi_process=False,
    )
    game_config = tictactoe_game_config
    config = NFSPDQNConfig(config_dict, game_config)
    env = game_config.make_env()

    trainer = NFSPTrainer(config=config, env=env, device=device)
    trainer.setup()

    initial_step = trainer.training_step
    trainer.train_step()

    assert trainer.training_step == initial_step + 1
