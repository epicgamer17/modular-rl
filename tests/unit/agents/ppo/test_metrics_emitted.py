import pytest
import torch
import numpy as np
from agents.ppo.agent import PPOAgent
from agents.ppo.config import PPOConfig
from runtime.context import ExecutionContext

pytestmark = pytest.mark.unit

def test_metrics_emitted():
    """
    Ensures that the PPO update_step returns all required diagnostic metrics.
    """
    obs_dim = 4
    act_dim = 2
    rollout_steps = 16
    
    config = PPOConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        rollout_steps=rollout_steps,
        num_envs=1,
        minibatch_size=8,
        epochs=1,
    )
    
    # We need an environment to initialize the agent
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    agent = PPOAgent(config, env)
    
    # Fill the buffer with dummy data
    for _ in range(rollout_steps):
        agent.actor_runtime.step(agent.ctx)
        
    # Run update_step
    metrics = agent.learner_runtime.update_step(agent.ctx)
    
    # Required keys
    required_keys = [
        "episodic_return",
        "episodic_length",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "sps"
    ]
    
    for key in required_keys:
        assert key in metrics, f"Metric '{key}' missing from update_step output. Found: {list(metrics.keys())}"
        assert isinstance(metrics[key], (int, float, np.float32, np.float64)), f"Metric '{key}' should be a number, got {type(metrics[key])}"

    print("\n[Success] All diagnostic metrics are present and correctly formatted.")

if __name__ == "__main__":
    test_metrics_emitted()
