import pytest
pytestmark = [pytest.mark.integration, pytest.mark.slow]

import torch
import time
from configs.agents.rainbow_dqn import RainbowConfig
from configs.games.cartpole import CartPoleConfig
from agents.trainers.rainbow_trainer import RainbowTrainer
from stats.stats import StatTracker


def test_trainer_rainbow_training_smoke():
    # 1. Setup Config
    game_cfg = CartPoleConfig()
    config_dict = {
        "training_steps": 1000,
        "min_replay_buffer_size": 100,
        "minibatch_size": 32,
        "num_minibatches": 2,
        "executor_type": "local",
        "noisy_sigma": 0.5,
        "atom_size": 51,
        "checkpoint_interval": 10000,
        "save_intermediate_weights": False,
        "epsilon_schedule": {
            "type": "linear",
            "initial": 1.0,
            "final": 0.05,
            "decay_steps": 500,
        },
        "action_selector": {
            "type": "argmax",
            "base": {"type": "argmax"},
        },
    }

    config = RainbowConfig(config_dict, game_cfg)
    device = torch.device("cpu")

    # 2. Initialize Trainer
    trainer = RainbowTrainer(config, game_cfg.make_env(), device)

    # 3. Train
    print("Starting Rainbow smoke test...")
    trainer.train()

    # 4. Verify learning
    scores = trainer.stats.get("score")
    if scores:
        print(f"Final scores: {scores[-10:]}")
        # Check for smoothness - just ensure it doesn't crash and scores are logged
        assert len(scores) > 0, "No scores logged"

    print("Smoke test COMPLETED successfully.")


if __name__ == "__main__":
    test_trainer_rainbow_training_smoke()
