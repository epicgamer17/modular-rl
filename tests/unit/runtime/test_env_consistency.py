import pytest
import torch
import gymnasium as gym
import numpy as np
from runtime.environment import wrap_env, StepResult

pytestmark = pytest.mark.unit

def test_gym_adapter_trace_matches_raw_env():
    """Verify that SingleToBatchEnvAdapter trace matches raw Gym env exactly over 20 steps."""
    seed = 42
    env_id = "CartPole-v1"
    
    # 1. Collect trace from raw Gym env
    raw_env = gym.make(env_id)
    raw_obs, _ = raw_env.reset(seed=seed)
    
    gym_obs_trace = [raw_obs]
    gym_reward_trace = []
    gym_done_trace = []
    
    # Use fixed actions for determinism
    actions = [0, 1, 0, 1, 0, 0, 1, 1, 0, 1] * 2 # 20 actions
    
    for a in actions:
        obs, reward, terminated, truncated, info = raw_env.step(a)
        gym_obs_trace.append(obs)
        gym_reward_trace.append(reward)
        gym_done_trace.append(terminated or truncated)
        if terminated or truncated:
            break
            
    raw_env.close()
    
    # 2. Collect trace from Adapter
    adapter_raw_env = gym.make(env_id)
    adapter = wrap_env(adapter_raw_env)
    
    adapter_obs = adapter.reset(seed=seed)
    
    # Adapter returns [1, obs_dim]
    np.testing.assert_allclose(adapter_obs[0].numpy(), gym_obs_trace[0], atol=1e-6)
    
    adapter_obs_trace = [adapter_obs[0].numpy()]
    adapter_reward_trace = []
    adapter_done_trace = []
    
    for i, a in enumerate(actions):
        if i >= len(gym_reward_trace):
            break # Match raw env length if it terminated early
            
        # Action must be batched [1]
        action_batch = torch.tensor([a], dtype=torch.long)
        step_res = adapter.step(action_batch)
        
        adapter_obs_trace.append(step_res.obs[0].numpy())
        adapter_reward_trace.append(step_res.reward[0].item())
        adapter_done_trace.append(step_res.terminated[0].item() or step_res.truncated[0].item())
        
    adapter.close()
    
    # 3. Compare traces
    assert len(adapter_obs_trace) == len(gym_obs_trace)
    for i in range(len(gym_obs_trace)):
        np.testing.assert_allclose(adapter_obs_trace[i], gym_obs_trace[i], atol=1e-6, err_msg=f"Obs mismatch at step {i}")
        
    assert len(adapter_reward_trace) == len(gym_reward_trace)
    for i in range(len(gym_reward_trace)):
        assert adapter_reward_trace[i] == gym_reward_trace[i], f"Reward mismatch at step {i}"
        
    assert len(adapter_done_trace) == len(gym_done_trace)
    for i in range(len(gym_done_trace)):
        assert adapter_done_trace[i] == gym_done_trace[i], f"Done mismatch at step {i}"
