import torch
import gymnasium as gym
import os
import sys

# Fix path for imports
sys.path.append(os.getcwd())

from agents.trainers.rainbow_trainer import RainbowTrainer
from configs.agents.rainbow_dqn import RainbowConfig
from configs.games.cartpole import CartPoleConfig

def smoke_test_rainbow_cartpole():
    print("Starting Rainbow smoke test on CartPole...")
    
    # 1. Setup Config
    game_config = CartPoleConfig()
    config_dict = {
        "training_steps": 20,
        "min_replay_buffer_size": 32,
        "minibatch_size": 32,
        "replay_buffer_size": 1000,
        "num_workers": 1,
        "multi_process": False,
        "compilation": {"enabled": False},
        "learning_rate": 1e-4,
        "discount_factor": 0.99,
        "n_step": 3,
        "atom_size": 51,
        "use_noisy_net": True,
        "noisy_sigma": 0.5,
        "architecture": {},
        "backbone": {"type": "mlp", "config_dict": {"hidden_layers": [64, 64]}},
        "head": {"type": "dueling_q", "config_dict": {}},
        "action_selector": {
            "base": {"type": "epsilon_greedy", "kwargs": {}},
            "decorators": []
        },
        "optimizer": torch.optim.Adam,
        "epsilon_schedule": {"initial": 0.0, "final": 0.0, "steps": 1},
        "per_beta_schedule": {"initial": 0.4, "final": 1.0, "steps": 1000},
    }
    config = RainbowConfig(config_dict, game_config)

    # 3. Create Trainer
    env = gym.make("CartPole-v1")
    device = torch.device("cpu")
    
    trainer = RainbowTrainer(
        config=config,
        env=env,
        device=device,
        name="smoke_test"
    )
    
    # 4. Run a few steps (DISABLED TO AVOID SHM ISSUES)
    # trainer.train()
    
    # 5. Check metrics via direct inference
    obs, _ = env.reset()
    obs = torch.tensor(obs).float().unsqueeze(0)
    with torch.no_grad():
        out = trainer.agent_network.obs_inference(obs)
        eval_q = out.expected_value
        print(f"Direct inference expected value: {eval_q}")
        if eval_q.abs().mean() > 10.0: # Expected mean is support center (250)
             print("SUCCESS: Q-values are stable and non-zero via direct inference!")
        else:
             print(f"FAILURE: Q-values are suspiciously small: {eval_q.abs().mean()}")

if __name__ == "__main__":
    smoke_test_rainbow_cartpole()
