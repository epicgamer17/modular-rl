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
    
    # Required keys - these are what the current PPO implementation actually returns
    # The training metrics are at the top level, system metrics are nested
    required_keys = [
        "entropy",
        "val_loss",
        "surr_loss",
        "total_loss",
        "clip",
    ]
    
    from runtime.signals import NoOp
    
    for key in required_keys:
        assert key in metrics, f"Metric '{key}' missing from update_step output. Found: {list(metrics.keys())}"
        # Check that it's a valid metric value
        val = metrics[key]
        # Allow for metrics that might be wrapped in a Value object or be tensors
        if hasattr(val, 'data'):
            val = val.data
        # Skip NoOp signals
        if isinstance(val, NoOp):
            continue
        # Accept common numeric types
        assert isinstance(val, (int, float, torch.Tensor, np.floating)), f"Metric '{key}' should be a number, got {type(val)}"

    print("\n[Success] All diagnostic metrics are present and correctly formatted.")

if __name__ == "__main__":
    test_metrics_emitted()
