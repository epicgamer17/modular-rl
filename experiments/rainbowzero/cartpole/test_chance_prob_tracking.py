
import sys
import os
import torch
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_configs.muzero_config import MuZeroConfig
from agents.muzero import MuZeroAgent
from modules.world_models.muzero_world_model import MuzeroWorldModel
from game_configs.cartpole_config import CartPoleConfig

def test_chance_prob_tracking():
    # 1. Setup Config
    game_config = CartPoleConfig()
    config_dict = {
        "world_model_cls": MuzeroWorldModel,
        "stochastic": True,
        "num_chance": 4,
        "value_loss_factor": 1.0, # minimal config
        "num_simulations": 2, # Speed up MCTS
        "minibatch_size": 2,
        "min_replay_buffer_size": 2,
        "initial_buffer_size": 2,
        "num_minibatches": 1,
        "training_steps": 1,
        "checkpoint_interval": 10,
        "test_interval": 100,
        "use_true_chance_codes": False, # Use predicted codes to populate stats
        "multi_process": False, # Run single process to avoid device issues
        # Network Config for Vector Env
        "residual_layers": [],
        "conv_layers": [],
        "dense_layer_widths": [64, 64],
        "representation_residual_layers": [],
        "representation_conv_layers": [],
        "representation_dense_layer_widths": [64, 64],
        "dynamics_residual_layers": [],
        "dynamics_conv_layers": [],
        "dynamics_dense_layer_widths": [64, 64],
        "reward_conv_layers": [],
        "reward_dense_layer_widths": [64],
        "to_play_conv_layers": [],
        "to_play_dense_layer_widths": [64],
        "critic_conv_layers": [],
        "critic_dense_layer_widths": [64],
        "actor_conv_layers": [],
        "actor_dense_layer_widths": [64],
        "afterstate_residual_layers": [],
        "afterstate_conv_layers": [],
        "afterstate_dense_layer_widths": [64, 64],
        "chance_conv_layers": [],
        "chance_dense_layer_widths": [64],
    }
    
    config = MuZeroConfig(config_dict, game_config)
    
    # 2. Setup Agent
    # Mock environment not strictly needed if we just step the training loop manually or rely on play_game
    # But MuZeroAgent needs an env to init.
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    
    # Mocking env.spec logic in agent init if needed, but CartPole-v1 usually works fine or we can duck type
    # Agent expects env.spec.reward_threshold sometimes.
    
    agent = MuZeroAgent(env, config, name="test_chance_prob")
    
    # 3. Simulate Data in Replay Buffer
    # We need to populate the replay buffer so that learn() can run.
    # play_game() populates the buffer.
    print("Playing games to populate buffer...")
    for _ in range(5):
        agent.play_game()
        
    print(f"Replay buffer size: {agent.replay_buffer.size}")
    
    # 4. Trigger Training
    # learn() is called inside train(), but we can call it manually if we mock the loop or just call train for 1 step
    # train() loop is a bit complex with checking buffer size. 
    # Let's just call agent.learn() manually since we have data.
    
    print("Running agent.learn()...")
    
    # set attributes normally set in train() loop
    agent.training_step = 0
    
    # Run learning step
    agent.learn()
    
    # 5. Verify Stats
    print("\nVerifying Stats...")
    stats = agent.stats.stats # direct access to dict
    
    missing_keys = []
    for i in range(config.num_chance):
        key = f"chance_prob_{i}"
        if key in stats:
            val = stats[key][-1].item() # get last value
            print(f"{key}: {val}")
            assert 0.0 <= val <= 1.0, f"Probability {val} out of range [0, 1]"
        else:
            missing_keys.append(key)
            
    if missing_keys:
        print(f"FAILED: Missing keys: {missing_keys}")
        sys.exit(1)
        
    print("SUCCESS: All chance prob keys found and valid.")
    
    # Clean up
    if os.path.exists(f"checkpoints/test_chance_prob"):
        shutil.rmtree(f"checkpoints/test_chance_prob")


if __name__ == "__main__":
    test_chance_prob_tracking()
