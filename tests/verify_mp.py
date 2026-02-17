import torch
import time
import os
from trainers.muzero_trainer import MuZeroTrainer
from configs.agents.muzero import MuZeroConfig
from configs.games.cartpole_config import CartPoleConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from losses.basic_losses import CategoricalCrossentropyLoss


def action_as_onehot(action, num_actions):
    if isinstance(action, torch.Tensor):
        action = action.item()
    one_hot = torch.zeros(num_actions)
    one_hot[action] = 1.0
    return one_hot


def verify():
    print("Starting verification of MuZero multiprocessing using MuZeroTrainer...")
    game_config = CartPoleConfig()
    env = game_config.make_env()

    config_dict = {
        "world_model_cls": MuzeroWorldModel,
        "residual_layers": [],
        "representation_dense_layer_widths": [64],
        "dynamics_dense_layer_widths": [64],
        "actor_dense_layer_widths": [16],
        "critic_dense_layer_widths": [16],
        "reward_dense_layer_widths": [16],
        "actor_conv_layers": [],
        "critic_conv_layers": [],
        "reward_conv_layers": [],
        "to_play_conv_layers": [],
        "num_simulations": 1,
        "minibatch_size": 1,
        "min_replay_buffer_size": 1,
        "replay_buffer_size": 100,
        "unroll_steps": 1,
        "n_step": 1,
        "multi_process": True,
        "num_workers": 1,
        "training_steps": 2,
        "games_per_generation": 1,
        "action_function": action_as_onehot,
        "value_loss_function": CategoricalCrossentropyLoss(),
        "reward_loss_function": CategoricalCrossentropyLoss(),
        "policy_loss_function": CategoricalCrossentropyLoss(),
        "support_range": 31,
        "model_name": "verify_mp_trainer",
    }

    config = MuZeroConfig(config_dict, game_config)

    # Use CPU for verification to avoid MPS/CUDA issues on smoke tests
    device = torch.device("cpu")

    print("Initializing Trainer...")
    trainer = MuZeroTrainer(config, env, device=device)

    # Disable checkpointing/plotting to avoid permission errors
    trainer._save_checkpoint = lambda: None

    try:
        print("Starting training with MP executor...")
        # Note: trainer.train() will start workers, collect data, and stop workers
        trainer.train()

        print("Training step:", trainer.training_step)
        assert trainer.training_step >= 1

        print("MuZeroTrainer MP Verification PASSED!")

    except Exception as e:
        print(f"Verification FAILED: {e}")
        import traceback

        traceback.print_exc()
        if hasattr(trainer, "executor") and trainer.executor:
            trainer.executor.stop()
        exit(1)


if __name__ == "__main__":
    verify()
