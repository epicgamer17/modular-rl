import pytest
pytestmark = [pytest.mark.integration, pytest.mark.slow]

import torch
import time
import os
from agents.trainers.muzero_trainer import MuZeroTrainer
from configs.agents.muzero import MuZeroConfig
from configs.games.cartpole import CartPoleConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
import torch.nn.functional as F


def action_as_onehot(action, num_actions):
    if isinstance(action, torch.Tensor):
        action = action.item()
    one_hot = torch.zeros(num_actions)
    one_hot[action] = 1.0
    return one_hot


import torch.multiprocessing as mp

try:
    import platform

    if platform.system() == "Darwin":
        mp.set_sharing_strategy("file_system")
except Exception:
    pass


def test_trainer_muzero_mp_verification(make_muzero_config_dict):
    print("Starting verification of MuZero multiprocessing using MuZeroTrainer...")
    game_config = CartPoleConfig()
    env = game_config.make_env()

    config_dict = make_muzero_config_dict(
        world_model_cls=MuzeroWorldModel,
        residual_layers=[],
        representation_dense_layer_widths=[64],
        dynamics_dense_layer_widths=[64],
        actor_dense_layer_widths=[16],
        critic_dense_layer_widths=[16],
        reward_dense_layer_widths=[16],
        actor_conv_layers=[],
        critic_conv_layers=[],
        reward_conv_layers=[],
        to_play_conv_layers=[],
        num_simulations=1,
        minibatch_size=1,
        min_replay_buffer_size=1,
        replay_buffer_size=100,
        unroll_steps=1,
        n_step=1,
        multi_process=True,
        num_workers=1,
        training_steps=2,
        games_per_generation=1,
        action_function=action_as_onehot,
        value_loss_function=F.cross_entropy,
        reward_loss_function=F.cross_entropy,
        policy_loss_function=F.cross_entropy,
        support_range=31,
        action_selector={"type": "argmax", "base": {"type": "argmax"}},
    )

    config = MuZeroConfig(config_dict, game_config)

    # Use CPU for verification to avoid MPS/CUDA issues on smoke tests
    device = torch.device("cpu")

    print("Initializing Trainer...")
    trainer = MuZeroTrainer(config, env, device=device, name="verify_mp_trainer")

    # Disable checkpointing/plotting to avoid permission errors
    trainer._save_checkpoint = lambda: None

    try:
        print("Starting training with MP executor...")
        # Note: trainer.train() will start workers, collect data, and stop workers
        trainer.train()

        print("Training step:", trainer.training_step)
        assert trainer.training_step >= 1

        # Check num_slots
        num_slots = trainer.executor.shared_pool.num_slots
        print(f"Pool size: {num_slots}")
        assert num_slots >= 64, f"Expected at least 64 slots, got {num_slots}"

        # All slots should have been released after train() stops the executor
        # Wait a bit for async cleanup
        time.sleep(1.0)
        free_slots = trainer.executor.shared_pool.free_slots.qsize()
        print(f"Free slots after train(): {free_slots} / {num_slots}")
        assert (
            free_slots == num_slots
        ), f"Slot leak detected! {num_slots - free_slots} slots not returned."

        print("MuZeroTrainer MP Verification PASSED!")

    except Exception as e:
        print(f"Verification FAILED: {e}")
        import traceback

        traceback.print_exc()
        if hasattr(trainer, "executor") and trainer.executor:
            trainer.executor.stop()
        exit(1)


if __name__ == "__main__":
    test_trainer_muzero_mp_verification()
