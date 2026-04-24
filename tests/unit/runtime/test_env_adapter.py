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

def test_validate_step_result():
    """Verify that validate_step_result catches shape and type mismatches."""
    from runtime.environment import validate_step_result
    
    batch_size = 4
    obs = torch.zeros((batch_size, 4))
    reward = torch.zeros(batch_size)
    terminated = torch.zeros(batch_size, dtype=torch.bool)
    truncated = torch.zeros(batch_size, dtype=torch.bool)
    info = [{}] * batch_size
    
    step_res = StepResult(obs, reward, terminated, truncated, info)
    
    # 1. Correct call
    validate_step_result(step_res, batch_size)
    
    # 2. Shape mismatch
    with pytest.raises(RuntimeError, match="shape mismatch"):
        validate_step_result(step_res, batch_size + 1)
        
    # 3. Type mismatch (reward)
    bad_reward_step_res = StepResult(obs, torch.zeros(batch_size, dtype=torch.long), terminated, truncated, info)
    with pytest.raises(RuntimeError, match="must be floating point"):
        validate_step_result(bad_reward_step_res, batch_size)

def test_runtime_raises_scalar_reward():
    """Verify that ActorRuntime raises RuntimeError if the environment returns a scalar reward (invalid batch)."""
    from runtime.runtime import ActorRuntime
    from core.graph import Graph
    
    class BadEnv:
        num_envs = 1
        observation_space = gym.spaces.Box(low=0, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)
        
        def step(self, action):
            # Return a StepResult that has B=2 while num_envs=1
            return StepResult(
                obs=torch.zeros((2, 4)), 
                reward=torch.zeros(2),
                terminated=torch.zeros(2, dtype=torch.bool),
                truncated=torch.zeros(2, dtype=torch.bool),
                info=[{}, {}]
            )
        def reset(self, seed=None):
            return torch.zeros((1, 4)), {}
        def close(self):
            pass
    
    env = BadEnv()
    actor_rt = ActorRuntime(Graph(), env)
    # Ensure it didn't wrap it (or if it did, it's still our BadEnv behavior)
    actor_rt.env = env # Force it
    
    actor_rt.current_obs = torch.zeros((1, 4))
    
    # Mock the graph execution to return a dummy action
    import runtime.runtime as runtime_module
    orig_execute = runtime_module.execute
    runtime_module.execute = lambda *args, **kwargs: {"actor": torch.tensor([0])}
    
    try:
        with pytest.raises(RuntimeError, match="shape mismatch"):
            actor_rt.step()
    finally:
        runtime_module.execute = orig_execute
