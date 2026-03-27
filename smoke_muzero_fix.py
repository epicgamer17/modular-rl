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

# Root Directory (current directory)
sys.path.append(os.getcwd())

from agents.trainers.muzero_trainer import MuZeroTrainer
from configs.agents.muzero import MuZeroConfig
from configs.games.tictactoe import TicTacToeConfig
from stats.stats import StatTracker

def main():
    print("Initializing MuZero Smoke Test for Policy Mass Loss Fix...")
    game_config = TicTacToeConfig()
    
    # Minimal config to quickly hit a training step
    params = {
        "training_steps": 2,
        "steps_per_epoch": 32, 
        "num_simulations": 5,
        "unroll_steps": 3,
        "batch_size": 16,
        "num_minibatches": 1,
        "replay_buffer_size": 1000,
        "min_buffer_size": 16,
        "learning_rate": 1e-3,
        "device": "cpu",
        "search_batch_size": 1,
        "executor_type": "local",
        "action_selector": {"base": {"type": "mcts"}},
        
        # Backbones
        "representation_backbone": {"type": "mlp", "widths": [64]},
        "dynamics_backbone": {"type": "mlp", "widths": [64]},
        "prediction_backbone": {"type": "mlp", "widths": [64]},
        
        # Heads
        "value_head": {},
        "policy_head": {},
        "reward_head": {},
        "to_play_head": {},
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
    trainer.checkpoint_interval = 10000
    trainer.test_interval = 10000

    
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
