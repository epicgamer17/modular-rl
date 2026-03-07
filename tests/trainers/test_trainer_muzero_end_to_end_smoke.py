import pytest
pytestmark = [pytest.mark.integration, pytest.mark.slow]

import torch
from agents.trainers.muzero_trainer import MuZeroTrainer
from configs.agents.muzero import MuZeroConfig
from configs.games.cartpole import CartPoleConfig
from configs.games.tictactoe import TicTacToeConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
import torch.nn.functional as F


def action_as_onehot(action, num_actions):
    """
    Encodes an action as a one-hot vector.
    """
    if isinstance(action, torch.Tensor):
        action = action.item()
    one_hot = torch.zeros(num_actions)
    one_hot[action] = 1.0
    return one_hot


def action_as_plane(action, num_actions, height, width):
    """
    Encodes an action as a plane (for CNNs).
    """
    if isinstance(action, torch.Tensor):
        action = action.item()
    plane = torch.zeros(num_actions, height, width)
    plane[action, :, :] = 1.0
    return plane


def test_muzero_cartpole_smoke(make_muzero_config_dict):
    """
    Smoke test for MuZero on CartPole.
    Ensures trainer can initialize and run a training step.
    """
    game_config = CartPoleConfig()
    env = game_config.make_env()

    config_dict = make_muzero_config_dict(
        world_model_cls=MuzeroWorldModel,
        representation_backbone={"type": "dense", "widths": [64]},
        dynamics_backbone={"type": "dense", "widths": [64]},
        prediction_backbone={"type": "dense", "widths": [64]},
        num_simulations=2,
        minibatch_size=2,
        min_replay_buffer_size=2,
        replay_buffer_size=100,
        unroll_steps=2,
        n_step=2,
        multi_process=False,
        training_steps=2,
        games_per_generation=1,
        learning_rate=0.001,
        action_function=action_as_onehot,
        value_loss_function=F.cross_entropy,
        reward_loss_function=F.cross_entropy,
        policy_loss_function=F.cross_entropy,
        support_range=31,
        action_selector={"base": {"type": "categorical"}},
    )

    config = MuZeroConfig(config_dict, game_config)

    trainer = MuZeroTrainer(
        config, env, device=torch.device("cpu"), name="smoke_test_cartpole"
    )

    # Disable checkpointing/plotting to avoid permission errors in this environment
    trainer._save_checkpoint = lambda: None

    assert trainer.agent_network is not None
    assert trainer.buffer is not None

    # Run a few steps of self-play and a learner step to verify the whole loop
    trainer.train()
    assert trainer.training_step >= 1


def test_muzero_tictactoe_smoke(make_muzero_config_dict):
    """
    Smoke test for MuZero on TicTacToe.
    """
    game_config = TicTacToeConfig()
    env = game_config.make_env()

    config_dict = make_muzero_config_dict(
        world_model_cls=MuzeroWorldModel,
        representation_backbone={
            "type": "resnet",
            "filters": [16, 16],
            "kernel_sizes": [3, 3],
            "strides": [1, 1],
        },
        dynamics_backbone={
            "type": "resnet",
            "filters": [16, 16],
            "kernel_sizes": [3, 3],
            "strides": [1, 1],
        },
        prediction_backbone={"type": "dense", "widths": [64]},
        num_simulations=2,
        minibatch_size=2,
        min_replay_buffer_size=2,
        replay_buffer_size=100,
        unroll_steps=2,
        n_step=3,
        multi_process=False,
        training_steps=2,
        games_per_generation=1,
        action_function=action_as_plane,
        value_loss_function=F.mse_loss,
        reward_loss_function=F.mse_loss,
        policy_loss_function=F.cross_entropy,
        support_range=None,
        action_selector={"base": {"type": "categorical"}},
    )

    config = MuZeroConfig(config_dict, game_config)

    trainer = MuZeroTrainer(
        config, env, device=torch.device("cpu"), name="smoke_test_tictactoe"
    )

    # Disable checkpointing/plotting
    trainer._save_checkpoint = lambda: None

    assert trainer.agent_network is not None
    assert trainer.buffer is not None

    # Verify trainer can run
    trainer.train()
    assert trainer.training_step >= 1


if __name__ == "__main__":
    test_muzero_cartpole_smoke()
    print("CartPole smoke test passed")
    test_muzero_tictactoe_smoke()
    print("TicTacToe smoke test passed")
    print("All smoke tests passed!")
