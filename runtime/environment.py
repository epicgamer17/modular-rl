from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
import numpy as np

@dataclass(frozen=True)
class StepResult:
    """
    Contract for environment step results.
    Always batched (B, ...). If single environment, B=1.
    """
    obs: torch.Tensor       # [B, ...]
    reward: torch.Tensor    # [B]
    terminated: torch.Tensor # [B] (bool)
    truncated: torch.Tensor  # [B] (bool)
    info: List[Dict[str, Any]] # length B

    def __post_init__(self) -> None:
        """Validate shapes and types of StepResult."""
        # Use B from obs as the batch dimension reference
        assert self.obs.ndim >= 1, f"Observation must have at least one dimension (batch), got {self.obs.ndim}"
        batch_size = self.obs.shape[0]

        assert self.reward.shape == (batch_size,), (
            f"Reward shape must be ({batch_size},), got {self.reward.shape}"
        )
        assert self.terminated.shape == (batch_size,), (
            f"Terminated shape must be ({batch_size},), got {self.terminated.shape}"
        )
        assert self.truncated.shape == (batch_size,), (
            f"Truncated shape must be ({batch_size},), got {self.truncated.shape}"
        )
        assert len(self.info) == batch_size, (
            f"Info list length must be {batch_size}, got {len(self.info)}"
        )

        assert self.terminated.dtype == torch.bool, (
            f"Terminated must be bool tensor, got {self.terminated.dtype}"
        )
        assert self.truncated.dtype == torch.bool, (
            f"Truncated must be bool tensor, got {self.truncated.dtype}"
        )

class SingleToBatchEnvAdapter:
    """
    Adapts a single (non-vectorized) environment to the batched StepResult contract.
    Returns batches of size B=1.
    """
    def __init__(self, env: Any):
        self.env = env
        self.num_envs = 1
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        """Reset the environment and return batched observation."""
        obs, info = self.env.reset(seed=seed)
        # obs shape: [...] -> [1, ...]
        return torch.from_numpy(obs).float().unsqueeze(0)

    def step(self, action: Any) -> StepResult:
        """Step the environment and return StepResult."""
        # action is expected to be batched [1, ...]
        if isinstance(action, (torch.Tensor, np.ndarray)) and action.shape[0] == 1:
            actual_action = action[0]
            if isinstance(actual_action, torch.Tensor):
                actual_action = actual_action.cpu().numpy()
        else:
            actual_action = action

        obs, reward, terminated, truncated, info = self.env.step(actual_action)
        
        return StepResult(
            obs=torch.from_numpy(obs).float().unsqueeze(0),
            reward=torch.tensor([reward], dtype=torch.float32),
            terminated=torch.tensor([terminated], dtype=torch.bool),
            truncated=torch.tensor([truncated], dtype=torch.bool),
            info=[info]
        )

    def close(self):
        self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)

def wrap_env(env: Any) -> Any:
    """Ensures the environment follows the batched contract."""
    # Check if it's already vectorized or has a step method that returns StepResult
    # For now, we'll be explicit: if it's not our VectorEnv, we wrap it.
    # In a more mature system, we'd check for a 'is_vectorized' property.
    from runtime.vector_env import VectorEnv
    if isinstance(env, VectorEnv):
        return env
    
    # If it has num_envs > 1, assume it's already batched (e.g. raw gym VectorEnv)
    # but we might still want to wrap it to ensure StepResult.
    if hasattr(env, "num_envs") and env.num_envs > 1:
        # TODO: Should we wrap existing VectorEnvs to ensure StepResult?
        # For now, let's assume VectorEnv is our primary entry point.
        return env

    return SingleToBatchEnvAdapter(env)
