from dataclasses import dataclass, field
from typing import Optional, Union

@dataclass(frozen=True)
class PPOConfig:
    """
    Configuration for PPO Agent with full hyperparameters.
    """
    # Environment & Model dimensions
    obs_dim: int
    act_dim: int
    hidden_dim: int = 64

    # PPO Specific Knobs
    total_steps: int = 1000000
    rollout_steps: int = 2048
    num_envs: int = 1
    epochs: int = 10
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    normalize_advantages: bool = True
    anneal_lr: bool = True
    value_clip: bool = True
    reward_norm: bool = False

    # Infrastructure handles
    model_handle: str = "ppo_net"
    optimizer_handle: str = "main_opt"
    buffer_id: str = "main"

    def __post_init__(self):
        """Validate the configuration."""
        assert self.total_steps > 0, f"total_steps must be positive, got {self.total_steps}"
        assert self.rollout_steps > 0, f"rollout_steps must be positive, got {self.rollout_steps}"
        assert self.num_envs > 0, f"num_envs must be positive, got {self.num_envs}"
        assert self.epochs > 0, f"epochs must be positive, got {self.epochs}"
        assert self.minibatch_size > 0, f"minibatch_size must be positive, got {self.minibatch_size}"
        
        # Invalid minibatch divisibility rejected
        total_batch_size = self.rollout_steps * self.num_envs
        assert total_batch_size % self.minibatch_size == 0, (
            f"Total batch size ({total_batch_size}) must be divisible by "
            f"minibatch_size ({self.minibatch_size})."
        )

        # Negative clip rejected
        assert self.clip_coef >= 0, f"clip_coef must be non-negative, got {self.clip_coef}"
        
        assert 0 <= self.gamma <= 1, f"gamma must be between 0 and 1, got {self.gamma}"
        assert 0 <= self.gae_lambda <= 1, f"gae_lambda must be between 0 and 1, got {self.gae_lambda}"
        assert self.learning_rate > 0, f"learning_rate must be positive, got {self.learning_rate}"
