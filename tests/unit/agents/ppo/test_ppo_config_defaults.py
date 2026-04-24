import pytest
from agents.ppo.config import PPOConfig

pytestmark = pytest.mark.unit

def test_ppo_config_sane_defaults():
    """Verify PPOConfig has sane defaults."""
    config = PPOConfig(obs_dim=4, act_dim=2)
    assert config.learning_rate == 3e-4
    assert config.gamma == 0.99
    assert config.clip_coef == 0.2
    assert config.minibatch_size == 64

def test_ppo_config_invalid_minibatch_divisibility():
    """Verify PPOConfig rejects minibatch_size that doesn't divide total batch size."""
    # total_batch = rollout_steps * num_envs = 2048 * 1 = 2048
    # 2048 % 63 != 0
    with pytest.raises(AssertionError, match="must be divisible by minibatch_size"):
        PPOConfig(obs_dim=4, act_dim=2, minibatch_size=63)

def test_ppo_config_negative_clip_rejected():
    """Verify PPOConfig rejects negative clip_coef."""
    with pytest.raises(AssertionError, match="clip_coef must be non-negative"):
        PPOConfig(obs_dim=4, act_dim=2, clip_coef=-0.1)

def test_ppo_config_valid_custom():
    """Verify PPOConfig accepts valid custom values."""
    config = PPOConfig(
        obs_dim=4, 
        act_dim=2, 
        rollout_steps=128, 
        num_envs=4, 
        minibatch_size=32
    )
    # total_batch = 128 * 4 = 512. 512 % 32 == 0.
    assert config.minibatch_size == 32
