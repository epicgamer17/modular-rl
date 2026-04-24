import pytest
import numpy as np
from runtime.vector_env import VectorEnv

pytestmark = pytest.mark.unit

def test_vec_env_shapes():
    """Verify that VectorEnv returns batches of correct shapes."""
    num_envs = 4
    env = VectorEnv("CartPole-v1", num_envs=num_envs)
    
    obs = env.reset(seed=42)
    
    # CartPole obs dim is 4
    assert obs.shape == (num_envs, 4)
    
    # Take a step
    actions = np.array([env.single_action_space.sample() for _ in range(num_envs)])
    step_res = env.step(actions)
    next_obs, rewards, terminations, truncations = step_res.obs, step_res.reward, step_res.terminated, step_res.truncated
    
    assert next_obs.shape == (num_envs, 4)
    assert rewards.shape == (num_envs,)
    assert terminations.shape == (num_envs,)
    assert truncations.shape == (num_envs,)
    
    env.close()

def test_async_vec_env_shapes():
    """Verify that Async VectorEnv returns batches of correct shapes."""
    num_envs = 2
    env = VectorEnv("CartPole-v1", num_envs=num_envs, async_mode=True)
    
    obs = env.reset(seed=42)
    assert obs.shape == (num_envs, 4)
    
    actions = np.array([env.single_action_space.sample() for _ in range(num_envs)])
    step_res = env.step(actions)
    next_obs, rewards = step_res.obs, step_res.reward
    
    assert next_obs.shape == (num_envs, 4)
    assert rewards.shape == (num_envs,)
    
    env.close()
