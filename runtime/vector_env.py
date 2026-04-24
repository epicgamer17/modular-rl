import gymnasium as gym
from typing import Callable, List, Optional, Any
import numpy as np
import torch


# TODO: should this be handled here? how do we also allow for PettingZoo or MuJoCo etc?
def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    """Helper to create a single environment instance."""

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env.action_space.seed(seed)
        return env

    return thunk


# TODO: allow for PufferLib Envs?
from runtime.environment import EnvAdapter, StepResult, ObsBatch
from core.schema import TensorSpec

class VectorEnv(EnvAdapter):
    """
    Wrapper around Gymnasium's VectorEnv to ensure consistent interface.
    Supports both Sync and Async variants.

    Accepts either an `env_id` (factory path: builds the underlying gym vector env)
    or a pre-built `envs` instance (adapter path: wraps it directly).
    """

    def __init__(
        self,
        env_id: Optional[str] = None,
        num_envs: int = 1,
        seed: int = 42,
        async_mode: bool = False,
        capture_video: bool = False,
        run_name: str = "default",
        envs: Optional[Any] = None,
    ):
        self.auto_reset = True

        if envs is not None:
            self.envs = envs
            self.num_envs = envs.num_envs
            spec = getattr(envs, "spec", None)
            self.env_id = spec.id if spec is not None else None
        else:
            if env_id is None:
                raise ValueError("VectorEnv requires either env_id or envs")
            self.num_envs = num_envs
            self.env_id = env_id

            env_fns = [
                make_env(env_id, seed + i, i, capture_video, run_name)
                for i in range(num_envs)
            ]

            if async_mode:
                self.envs = gym.vector.AsyncVectorEnv(env_fns)
            else:
                self.envs = gym.vector.SyncVectorEnv(env_fns)

        # Populate specs. Gym vector envs expose single_{obs,action}_space;
        # fall back to {observation,action}_space for non-gym-vector inputs.
        obs_space = getattr(self.envs, "single_observation_space", None) or self.envs.observation_space
        act_space = getattr(self.envs, "single_action_space", None) or self.envs.action_space

        self.obs_spec = TensorSpec(shape=obs_space.shape, dtype=str(obs_space.dtype))

        if hasattr(act_space, "n"):
            # Discrete
            self.act_spec = TensorSpec(shape=(), dtype="int64")
        else:
            # Continuous
            self.act_spec = TensorSpec(shape=act_space.shape, dtype=str(act_space.dtype))

        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self, seed: Optional[int] = None) -> ObsBatch:
        """Reset all environments and return batched observations."""
        obs, info = self.envs.reset(seed=seed)
        # Normalize to float32 to prevent float64 leaks
        return torch.from_numpy(obs).to(torch.float32)

    def step(self, actions: Any) -> 'StepResult':
        """Step all environments with a batch of actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        # Normalize actions to env-native dtype
        if hasattr(self.action_space, "dtype"):
            native_dtype = self.action_space.dtype
            if isinstance(actions, np.ndarray):
                actions = actions.astype(native_dtype)

        obs, reward, terminated, truncated, info = self.envs.step(actions)
        
        from runtime.environment import StepResult
        
        # Convert info (dict of arrays) to list of dicts
        if isinstance(info, dict):
            # Gym VectorEnv often returns info as a dict of arrays
            # We need to reconstruct it into a list of dicts
            batch_info = []
            for i in range(self.num_envs):
                env_info = {}
                for k, v in info.items():
                    # Handle both array-like and list-like info values
                    if hasattr(v, "__getitem__"):
                        try:
                            env_info[k] = v[i]
                        except (IndexError, KeyError):
                            # Fallback for keys that might not be batched or have different structure
                            pass
                batch_info.append(env_info)
        else:
            batch_info = info if isinstance(info, list) else [info] * self.num_envs

        return StepResult(
            obs=torch.from_numpy(obs).to(torch.float32),
            reward=torch.from_numpy(reward).to(torch.float32),
            terminated=torch.from_numpy(terminated).bool(),
            truncated=torch.from_numpy(truncated).bool(),
            info=batch_info
        )

    def close(self):
        """Close all environments."""
        self.envs.close()

    @property
    def single_observation_space(self):
        return self.envs.single_observation_space

    @property
    def single_action_space(self):
        return self.envs.single_action_space

    def __getattr__(self, name):
        if name == "envs":
            raise AttributeError(name)
        return getattr(self.envs, name)
