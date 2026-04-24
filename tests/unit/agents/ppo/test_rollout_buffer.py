import pytest
import torch
from agents.ppo.buffer import RolloutBuffer

pytestmark = pytest.mark.unit

def test_rollout_buffer_capacity():
    """Verify that RolloutBuffer stores exactly rollout_steps * num_envs transitions."""
    rollout_steps = 10
    num_envs = 4
    obs_dim = 4
    buffer = RolloutBuffer(rollout_steps, num_envs, obs_dim)
    
    # Fill the buffer
    for _ in range(rollout_steps):
        buffer.add(
            obs=torch.zeros((num_envs, obs_dim)),
            action=torch.zeros(num_envs),
            reward=torch.zeros(num_envs),
            terminated=torch.zeros(num_envs),
            truncated=torch.zeros(num_envs),
            value=torch.zeros(num_envs),
            log_prob=torch.zeros(num_envs)
        )
    
    assert len(buffer) == rollout_steps * num_envs
    assert buffer.full is True
    
    # Try adding one more, should fail
    with pytest.raises(AssertionError, match="Buffer is full"):
        buffer.add(
            obs=torch.zeros((num_envs, obs_dim)),
            action=torch.zeros(num_envs),
            reward=torch.zeros(num_envs),
            terminated=torch.zeros(num_envs),
            truncated=torch.zeros(num_envs),
            value=torch.zeros(num_envs),
            log_prob=torch.zeros(num_envs)
        )

def test_rollout_buffer_clear():
    """Verify that RolloutBuffer empties after clear()."""
    rollout_steps = 5
    num_envs = 2
    obs_dim = 4
    buffer = RolloutBuffer(rollout_steps, num_envs, obs_dim)
    
    buffer.add(
        obs=torch.zeros((num_envs, obs_dim)),
        action=torch.zeros(num_envs),
        reward=torch.zeros(num_envs),
        terminated=torch.zeros(num_envs),
        truncated=torch.zeros(num_envs),
        value=torch.zeros(num_envs),
        log_prob=torch.zeros(num_envs)
    )
    
    assert len(buffer) == num_envs
    buffer.clear()
    assert len(buffer) == 0
    assert buffer.ptr == 0
