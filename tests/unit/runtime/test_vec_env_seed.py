import pytest
import numpy as np
from runtime.io.vector_env import VectorEnv

pytestmark = pytest.mark.unit

def test_vec_env_determinism():
    """Verify that VectorEnv is deterministic with fixed seed."""
    num_envs = 4
    seed = 42
    
    # Create first env
    env1 = VectorEnv("CartPole-v1", num_envs=num_envs, seed=seed)
    obs1 = env1.reset(seed=seed)
    actions1 = np.array([env1.single_action_space.sample() for _ in range(num_envs)])
    step_res1 = env1.step(actions1)
    next_obs1, rewards1 = step_res1.obs, step_res1.reward
    
    # Create second env with same seed
    env2 = VectorEnv("CartPole-v1", num_envs=num_envs, seed=seed)
    obs2 = env2.reset(seed=seed)
    actions2 = np.array([env2.single_action_space.sample() for _ in range(num_envs)])
    step_res2 = env2.step(actions2)
    next_obs2, rewards2 = step_res2.obs, step_res2.reward
    
    # Assert they are the same
    np.testing.assert_array_equal(obs1, obs2)
    np.testing.assert_array_equal(actions1, actions2)
    np.testing.assert_array_equal(next_obs1, next_obs2)
    np.testing.assert_array_equal(rewards1, rewards2)
    
    env1.close()
    env2.close()

def test_vec_env_different_seeds():
    """Verify that VectorEnv is different with different seeds."""
    num_envs = 2
    
    env1 = VectorEnv("CartPole-v1", num_envs=num_envs, seed=42)
    obs1 = env1.reset(seed=42)
    
    env2 = VectorEnv("CartPole-v1", num_envs=num_envs, seed=123)
    obs2 = env2.reset(seed=123)
    
    # Very unlikely to be equal
    assert not np.array_equal(obs1, obs2)
    
    env1.close()
    env2.close()
