from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
import numpy as np
import gymnasium as gym
from abc import ABC, abstractmethod
from core.schema import TensorSpec

# Type alias for batched observations
ObsBatch = torch.Tensor


@dataclass(frozen=True)
class StepResult:
    """
    Contract for environment step results.
    Always batched (B, ...). If single environment, B=1.
    """

    obs: ObsBatch  # [B, ...]
    reward: torch.Tensor  # [B]
    terminated: torch.Tensor  # [B] (bool)
    truncated: torch.Tensor  # [B] (bool)
    info: List[Dict[str, Any]]  # length B

    def __post_init__(self) -> None:
        """Validate shapes and types of StepResult."""
        # Use B from obs as the batch dimension reference
        assert (
            self.obs.ndim >= 1
        ), f"Observation must have at least one dimension (batch), got {self.obs.ndim}"
        batch_size = self.obs.shape[0]

        assert self.reward.shape == (
            batch_size,
        ), f"Reward shape must be ({batch_size},), got {self.reward.shape}"
        assert self.terminated.shape == (
            batch_size,
        ), f"Terminated shape must be ({batch_size},), got {self.terminated.shape}"
        assert self.truncated.shape == (
            batch_size,
        ), f"Truncated shape must be ({batch_size},), got {self.truncated.shape}"
        assert (
            len(self.info) == batch_size
        ), f"Info list length must be {batch_size}, got {len(self.info)}"

        assert (
            self.terminated.dtype == torch.bool
        ), f"Terminated must be bool tensor, got {self.terminated.dtype}"
        assert (
            self.truncated.dtype == torch.bool
        ), f"Truncated must be bool tensor, got {self.truncated.dtype}"


def validate_step_result(step_res: StepResult, num_envs: int) -> None:
    """
    Runtime validation of StepResult against expected num_envs.
    Raises RuntimeError if any contract is violated.
    """
    # 1. Shape Checks
    if step_res.reward.shape != (num_envs,):
        raise RuntimeError(
            f"StepResult.reward shape mismatch. Expected ({num_envs},), got {step_res.reward.shape}"
        )

    if step_res.terminated.shape != (num_envs,):
        raise RuntimeError(
            f"StepResult.terminated shape mismatch. Expected ({num_envs},), got {step_res.terminated.shape}"
        )

    if step_res.truncated.shape != (num_envs,):
        raise RuntimeError(
            f"StepResult.truncated shape mismatch. Expected ({num_envs},), got {step_res.truncated.shape}"
        )

    # 2. Type Checks
    if not torch.is_floating_point(step_res.reward):
        raise RuntimeError(
            f"StepResult.reward must be floating point, got {step_res.reward.dtype}"
        )

    if step_res.terminated.dtype != torch.bool:
        raise RuntimeError(
            f"StepResult.terminated must be bool, got {step_res.terminated.dtype}"
        )

    if step_res.truncated.dtype != torch.bool:
        raise RuntimeError(
            f"StepResult.truncated must be bool, got {step_res.truncated.dtype}"
        )


class EnvAdapter(ABC):
    """
    Canonical interface for environment interaction.
    Always batched (B, ...).
    """

    num_envs: int
    obs_spec: TensorSpec
    act_spec: TensorSpec
    auto_reset: bool = False

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> ObsBatch:
        """Reset all environments and return batched observations."""
        pass

    @abstractmethod
    def step(self, action_batch: Any) -> StepResult:
        """Step all environments and return StepResult."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close environments."""
        pass


class SingleToBatchEnvAdapter(EnvAdapter):
    """
    Adapts a single (non-vectorized) environment to the batched StepResult contract.
    Returns batches of size B=1.
    """

    def __init__(self, env: Any):
        self.env = env
        self.num_envs = 1
        self.auto_reset = False

        # Convert gym spaces to TensorSpec
        obs_shape = env.observation_space.shape
        obs_dtype = str(env.observation_space.dtype)
        self.obs_spec = TensorSpec(shape=obs_shape, dtype=obs_dtype)

        if hasattr(env.action_space, "n"):
            # Discrete
            self.act_spec = TensorSpec(shape=(), dtype="int64")
        else:
            # Continuous
            self.act_spec = TensorSpec(
                shape=env.action_space.shape, dtype=str(env.action_space.dtype)
            )

    def reset(self, seed: Optional[int] = None) -> ObsBatch:
        """Reset the environment and return batched observation."""
        obs, info = self.env.reset(seed=seed)
        # obs shape: [...] -> [1, ...]
        # Normalize to float32 to prevent float64 leaks
        return torch.from_numpy(obs).to(torch.float32).unsqueeze(0)

    def step(self, action: Any) -> StepResult:
        """Step the environment and return StepResult."""
        # Unbatch action for single env (expecting batch size 1)
        if isinstance(action, (torch.Tensor, np.ndarray)):
            actual_action = action[0] if action.ndim > 0 else action
        else:
            actual_action = action

        # Normalize to env-native type
        if hasattr(self.env.action_space, "n"):
            # Discrete actions must be scalars (int)
            actual_action = int(np.asarray(actual_action).item())
        elif hasattr(self.env.action_space, "dtype"):
            # Continuous/Box actions should match space dtype
            native_dtype = self.env.action_space.dtype
            if isinstance(actual_action, torch.Tensor):
                actual_action = actual_action.cpu().numpy()
            actual_action = np.array(actual_action, dtype=native_dtype)

        obs, reward, terminated, truncated, info = self.env.step(actual_action)

        return StepResult(
            obs=torch.from_numpy(obs).to(torch.float32).unsqueeze(0),
            reward=torch.tensor([reward], dtype=torch.float32),
            terminated=torch.tensor([terminated], dtype=torch.bool),
            truncated=torch.tensor([truncated], dtype=torch.bool),
            info=[info],
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
    if isinstance(env, EnvAdapter):
        return env

    # If it has num_envs > 1, assume it's already batched (e.g. raw gym VectorEnv)
    if hasattr(env, "num_envs") and env.num_envs > 1:
        # TODO: Should we wrap existing VectorEnvs to ensure StepResult?
        # For now, let's assume VectorEnv is our primary entry point.
        return env

    return SingleToBatchEnvAdapter(env)
