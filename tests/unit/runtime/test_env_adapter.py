import pytest
import torch
import gymnasium as gym
import numpy as np
from runtime.environment import StepResult, SingleToBatchEnvAdapter, wrap_env
from runtime.vector_env import VectorEnv

pytestmark = pytest.mark.unit

def test_single_env_returns_batch1():
    """Verify that a single env wrapped by SingleToBatchEnvAdapter returns batched results (B=1)."""
    raw_env = gym.make("CartPole-v1")
    env = wrap_env(raw_env)
    
    # 1. Test reset
    obs = env.reset(seed=42)
    assert isinstance(obs, torch.Tensor), "Reset must return a torch.Tensor"
    assert obs.ndim == 2, f"Obs should have 2 dimensions [B, D], got {obs.ndim}"
    assert obs.shape[0] == 1, f"Batch size should be 1, got {obs.shape[0]}"
    
    # 2. Test step
    # Action for CartPole is discrete {0, 1}
    action = torch.tensor([0], dtype=torch.long)
    step_res = env.step(action)
    
    assert isinstance(step_res, StepResult)
    assert step_res.obs.shape[0] == 1
    assert step_res.reward.shape == (1,)
    assert step_res.terminated.shape == (1,)
    assert step_res.truncated.shape == (1,)
    assert len(step_res.info) == 1
    
    # Verify types
    assert step_res.terminated.dtype == torch.bool
    assert step_res.truncated.dtype == torch.bool
    
    env.close()

def test_vector_env_returns_batchN():
    """Verify that VectorEnv returns batched results of size N."""
    num_envs = 4
    env = VectorEnv(env_id="CartPole-v1", num_envs=num_envs, seed=42)
    
    # 1. Test reset
    obs = env.reset(seed=42)
    assert isinstance(obs, torch.Tensor)
    assert obs.shape[0] == num_envs, f"Expected batch size {num_envs}, got {obs.shape[0]}"
    
    # 2. Test step
    actions = np.zeros(num_envs, dtype=np.int64)
    step_res = env.step(actions)
    
    assert isinstance(step_res, StepResult)
    assert step_res.obs.shape[0] == num_envs
    assert step_res.reward.shape == (num_envs,)
    assert step_res.terminated.shape == (num_envs,)
    assert step_res.truncated.shape == (num_envs,)
    assert len(step_res.info) == num_envs
    
    # Verify types
    assert step_res.terminated.dtype == torch.bool
    assert step_res.truncated.dtype == torch.bool
    
    env.close()

def test_env_adapter_specs():
    """Verify that EnvAdapters populate obs_spec and act_spec correctly."""
    raw_env = gym.make("CartPole-v1")
    env = wrap_env(raw_env)
    
    assert hasattr(env, "obs_spec")
    assert hasattr(env, "act_spec")
    assert env.obs_spec.shape == (4,) # CartPole obs dim
    assert env.act_spec.dtype == "int64"
    
    env.close()
