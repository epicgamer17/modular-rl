import pytest
import torch
import numpy as np
import gymnasium as gym
from runtime.environment import SingleToBatchEnvAdapter, EnvWrapper, StepResult

pytestmark = pytest.mark.unit

def test_gym_wrapper_composition():
    """Verify that SingleToBatchEnvAdapter works with wrapped Gymnasium environments."""
    # Use a real env to ensure all attributes exist
    raw_env = gym.make("CartPole-v1")
    
    # Wrap it using a standard gym wrapper
    class gym_ClipReward(gym.RewardWrapper):
        def reward(self, reward):
            return np.clip(reward, -1, 1)

    raw_env = gym.make("CartPole-v1")
    wrapped_env = gym_ClipReward(raw_env)
    
    adapter = SingleToBatchEnvAdapter(wrapped_env)
    adapter.reset()
    res = adapter.step(torch.tensor([0]))
    
    # Reward should be clipped to 1.0 by the gym wrapper
    assert res.reward.item() == 1.0

def test_adapter_middleware_composition():
    """Verify that EnvWrapper can be used to create adapter middleware."""
    class MockAdapter(SingleToBatchEnvAdapter):
        def __init__(self):
            # We need a real-ish env for specs
            env = gym.make("CartPole-v1")
            super().__init__(env)
        def step(self, action):
            # Return a fixed result with high reward
            return StepResult(
                obs=torch.zeros((1, 4)),
                reward=torch.tensor([100.0]),
                terminated=torch.tensor([False]),
                truncated=torch.tensor([False]),
                info=[{}]
            )

    class ClipRewardAdapter(EnvWrapper):
        def step(self, action_batch):
            res = self.adapter.step(action_batch)
            # Transform the result
            return StepResult(
                obs=res.obs,
                reward=torch.clamp(res.reward, -1.0, 1.0),
                terminated=res.terminated,
                truncated=res.truncated,
                info=res.info
            )

    base_adapter = MockAdapter()
    wrapped_adapter = ClipRewardAdapter(base_adapter)
    
    res = wrapped_adapter.step(torch.tensor([0]))
    
    # Reward should be clipped to 1.0 by the adapter wrapper
    assert res.reward.item() == 1.0
    assert wrapped_adapter.num_envs == 1

def test_wrapper_preserves_contract():
    """Verify that a wrapped adapter still follows all StepResult invariants."""
    from runtime.environment import validate_step_result
    
    class IdentityWrapper(EnvWrapper):
        pass

    env = gym.make("CartPole-v1")
    adapter = SingleToBatchEnvAdapter(env)
    wrapped = IdentityWrapper(adapter)
    
    wrapped.reset()
    res = wrapped.step(torch.tensor([0]))
    
    # Should pass runtime validation
    validate_step_result(res, wrapped.num_envs)
    assert isinstance(res, StepResult)
