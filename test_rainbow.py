import gymnasium as gym
import torch
import sys

from agents.trainers.rainbow_trainer import RainbowTrainer
from configs.games.cartpole import CartPoleConfig
from configs.agents.rainbow_dqn import RainbowConfig
from stats.stats import StatTracker

env = gym.make("CartPole-v1")

config_dict = {
    "executor_type": "local",
    "adam_epsilon": 1e-8,
    "learning_rate": 0.001,
    "training_steps": 200,
    "minibatch_size": 32,
    "transfer_interval": 100,
    "n_step": 2,
    "replay_interval": 1,
    "kernel_initializer": "orthogonal",
    "clipnorm": 10.0,
    "model_name": "rainbow_smoke_test",
    "noisy_sigma": 0.5,
    "backbone": {
        "type": "dense",
        "widths": [32],
    },
    "head": {
        "output_strategy": {
            "type": "c51",
            "num_atoms": 51,
            "v_min": 0,
            "v_max": 200,
        },
        "value_hidden_widths": [],
        "advantage_hidden_widths": [],
    },
    "per_epsilon": 1e-6,
    "per_alpha": 0.2,
    "per_beta": 0.6,
    "action_selector": {
        "base": {
            "type": "argmax",
            "kwargs": {},
        },
    },
}

game_config = CartPoleConfig()
config = RainbowConfig(config_dict, game_config)
trainer = RainbowTrainer(
    config,
    env,
    torch.device("cpu"),
    "rainbow_refactor",
    StatTracker("rainbow_refactor"),
)

# Disable checkpointing/testing for smoke test
trainer.checkpoint_interval = 1000
trainer.test_interval = 1000

# Monkey patch train to print loss
original_step = trainer.learner.step


def mocked_step(stats):
    res = original_step(stats)
    if trainer.training_step % 10 == 0:
        print(f"Step {trainer.training_step}, Loss: {res.get('td_loss', 'N/A')}")
    return res


trainer.learner.step = mocked_step

print("Starting Rainbow training...")
trainer.train()
print("Rainbow training completed successfully.")
