import sys
import os
import torch
import numpy as np

# AVOID CIRCULAR IMPORTS: Import gymnasium and pettingzoo early
try:
    import gymnasium as gym
    import pettingzoo
except ImportError:
    pass

# Ensure we can import from the project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agents.trainers.muzero_trainer import MuZeroTrainer
from configs.agents.muzero import MuZeroConfig
from configs.games.tictactoe import TicTacToeConfig
from stats.stats import StatTracker

def main():
    print("Initializing MuZero Smoke Test for Policy Mass Loss Fix...")
    game_config = TicTacToeConfig()
    
    # Minimal config to quickly hit a training step
    params = {
        "training_steps": 5,
        "steps_per_epoch": 32, # Ensure we have enough data for a minibatch
        "num_simulations": 5,
        "unroll_steps": 3,
        "batch_size": 16,
        "num_minibatches": 1,
        "replay_buffer_size": 1000,
        "min_buffer_size": 16,
        "learning_rate": 1e-3,
        "device": "cpu",
        "search_batch_size": 1,
    }
    
    # Create config
    config = MuZeroConfig(config_dict=params, game_config=game_config)
    
    # Create env via factory
    env = game_config.env_factory()
    
    # Initialize trainer
    tracker = StatTracker(name="muzero_fix_verify")
    trainer = MuZeroTrainer(
        config=config,
        env=env,
        device=torch.device("cpu"),
        name="muzero_fix_verify",
        stats=tracker
    )
    
    trainer.test_agents = [] # Skip evaluation for speed
    
    print("Starting training loop...")
    try:
        # We only need one training step to verify if the assertion triggers
        trainer.train()
        print("\nSUCCESS: Training completed without Policy Mass Loss AssertionError!")
    except AssertionError as e:
        print(f"\nFAILURE: AssertionError triggered: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nFAILURE: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
