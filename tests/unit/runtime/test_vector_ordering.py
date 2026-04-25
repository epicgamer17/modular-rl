import pytest
import numpy as np
import gymnasium as gym
import torch
from runtime.io.vector_env import VectorEnv

pytestmark = pytest.mark.unit

class IDEnv(gym.Env):
    """Env that returns its assigned ID in its observation."""
    def __init__(self, env_id=0):
        super().__init__()
        self.env_id = env_id
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([float(self.env_id)], dtype=np.float32), {}
        
    def step(self, action):
        return np.array([float(self.env_id)], dtype=np.float32), 0.0, False, False, {}

# Register the env
try:
    gym.register(
        id='IDEnv-v0',
        entry_point=IDEnv,
    )
except gym.error.Error:
    pass # Already registered

def test_vector_ordering_guarantee():
    """Verify that VectorEnv (Sync and Async) preserves environment ordering."""
    num_envs = 8
    
    # We need to pass the env_id to each instance. 
    # Gymnasium's VectorEnv uses a factory function for each env.
    # Our VectorEnv.make_env uses gym.make(env_id).
    # To pass different IDs, we'd need to customize make_env.
    
    # Instead of modifying VectorEnv now, let's use a wrapper that adds the ID
    # based on the seed or some other identifiable info if possible.
    # Or we can just monkeypatch gym.make for this test.
    
    env_instances = []
    def mocked_make(id, **kwargs):
        idx = len(env_instances)
        env = IDEnv(env_id=idx)
        env_instances.append(env)
        return env
        
    import runtime.io.vector_env as ve_module
    orig_make_env = ve_module.make_env
    
    # Custom make_env that uses our mocked_make
    def custom_make_env(env_id, seed, idx, capture_video, run_name):
        def thunk():
            return IDEnv(env_id=idx)
        return thunk
        
    ve_module.make_env = custom_make_env
    
    try:
        for async_mode in [False, True]:
            env = VectorEnv("IDEnv-v0", num_envs=num_envs, async_mode=async_mode)
            
            obs = env.reset(seed=42)
            
            # Verify initial obs
            for i in range(num_envs):
                assert obs[i].item() == i, f"Initial obs mismatch for env {i} (async={async_mode})"
                
            # Step for 100 steps
            for step in range(100):
                actions = np.zeros(num_envs, dtype=np.int64)
                step_res = env.step(actions)
                
                for i in range(num_envs):
                    assert step_res.obs[i].item() == i, f"Obs mismatch at step {step} for env {i} (async={async_mode})"
            
            env.close()
    finally:
        ve_module.make_env = orig_make_env

if __name__ == "__main__":
    test_vector_ordering_guarantee()
