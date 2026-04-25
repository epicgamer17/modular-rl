import pytest
import torch
import numpy as np
import gymnasium as gym
from runtime.io.environment import SingleToBatchEnvAdapter
from runtime.io.vector_env import VectorEnv

pytestmark = pytest.mark.unit

def test_obs_dtype_float32():
    """Verify that observations are normalized to float32 even if the env returns float64."""
    class Float64Env(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float64)
            self.action_space = gym.spaces.Discrete(2)
        def reset(self, seed=None):
            return np.array([0.1, 0.2], dtype=np.float64), {}
        def step(self, action):
            return np.array([0.3, 0.4], dtype=np.float64), 1.0, False, False, {}

    env = Float64Env()
    adapter = SingleToBatchEnvAdapter(env)
    
    # Test reset
    obs = adapter.reset()
    assert obs.dtype == torch.float32, f"Expected float32, got {obs.dtype}"
    
    # Test step
    res = adapter.step(torch.tensor([0]))
    assert res.obs.dtype == torch.float32, f"Expected float32, got {res.obs.dtype}"
    assert res.reward.dtype == torch.float32, f"Expected float32, got {res.reward.dtype}"

def test_action_normalization_discrete():
    """Verify that actions are converted to env-native int types."""
    class IntActionEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
            self.action_space = gym.spaces.Discrete(2) # dtype is int64 by default in newer gym
        def reset(self, seed=None):
            return np.array([0.0]), {}
        def step(self, action):
            # If we pass a float or wrong type, some envs might crash or behave weirdly.
            # Here we just verify the type of the received action.
            assert isinstance(action, (int, np.integer)), f"Expected int, got {type(action)}"
            if hasattr(action, "dtype"):
                assert action.dtype == self.action_space.dtype
            return np.array([0.0]), 0.0, False, False, {}

    env = IntActionEnv()
    adapter = SingleToBatchEnvAdapter(env)
    adapter.reset()
    
    # Pass a float tensor action - should be converted to int64
    adapter.step(torch.tensor([1.0], dtype=torch.float32))

def test_vector_env_dtype_normalization():
    """Verify VectorEnv also normalizes dtypes."""
    # We'll use CartPole as it's standard, but it usually returns float32 anyway.
    # To really test normalization, we'd need an env returning float64.
    # But we can at least verify that VectorEnv returns float32.
    num_envs = 2
    env = VectorEnv("CartPole-v1", num_envs=num_envs)
    
    obs = env.reset()
    assert obs.dtype == torch.float32
    
    res = env.step(torch.zeros(num_envs, dtype=torch.int64))
    assert res.obs.dtype == torch.float32
    assert res.reward.dtype == torch.float32
    
    env.close()
