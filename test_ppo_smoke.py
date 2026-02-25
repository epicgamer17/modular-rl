import sys
import os
from agents.trainers.ppo_trainer import PPOTrainer
import gymnasium as gym
import torch
from configs.agents.ppo import PPOConfig
from configs.games.cartpole import CartPoleConfig
from stats.stats import StatTracker

env = gym.make("CartPole-v1", render_mode="rgb_array")
config_dict = {
    "model_name": "ppo_smoke_test",
    "executor_type": "local",
    "architecture": {"activation": "tanh", "kernel_initializer": "orthogonal"},
    "clip_param": 0.2, "discount_factor": 0.99, "gae_lambda": 0.95,
    "steps_per_epoch": 64, "train_policy_iterations": 1, "train_value_iterations": 1,
    "target_kl": 0.02, "entropy_coefficient": 0.01, "num_minibatches": 1,
    "training_steps": 150, "checkpoint_interval": 100, "test_interval": 60, "test_trials": 2,
    "policy_head": {"neck": {"type": "dense", "widths": [64]}},
    "value_head": {"neck": {"type": "dense", "widths": [64]}},
    "actor_config": {"learning_rate": 2.5e-4, "adam_epsilon": 1e-7, "clipnorm": 0.5},
    "critic_config": {"learning_rate": 2.5e-4, "adam_epsilon": 1e-7, "clipnorm": 0.5},
    "action_selector": {
        "base": {"type": "categorical"},
        "decorators": [{"type": "ppo_injector", "kwargs": {}}]
    }
}
config = PPOConfig(config_dict, CartPoleConfig())

# Override checkpoint saving logic for test
class TestPPOTrainer(PPOTrainer):
    def _save_checkpoint(self, checkpoint_data=None):
        if checkpoint_data is None:
            checkpoint_data = {
                "agent_network": self.agent_network.state_dict(),
                "policy_optimizer": self.learner.policy_optimizer.state_dict(),
                "value_optimizer": self.learner.value_optimizer.state_dict(),
                "policy_scheduler": self.learner.policy_scheduler.state_dict(),
                "value_scheduler": self.learner.value_scheduler.state_dict(),
            }
        base_dir = "/tmp/ppo_smoke_checkpoints/" + self.name
        step_dir = os.path.join(base_dir, f"step_{self.training_step}")
        os.makedirs(step_dir, exist_ok=True)
        # We don't need to actually save for this smoke test
        print(f"Skipping save to {step_dir} for test")

trainer = TestPPOTrainer(config, env, torch.device("cpu"), "ppo_smoke_test")
# Disable plotting to avoid font cache hang
trainer.stats.plot_graphs = lambda dir: None
trainer.train()
