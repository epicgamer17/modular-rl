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
    """

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        seed: int = 42,
        async_mode: bool = False,
        capture_video: bool = False,
        run_name: str = "default",
    ):
        self.num_envs = num_envs
        self.env_id = env_id
        self.auto_reset = True

        env_fns = [
            make_env(env_id, seed + i, i, capture_video, run_name)
            for i in range(num_envs)
        ]

        if async_mode:
            self.envs = gym.vector.AsyncVectorEnv(env_fns)
        else:
            self.envs = gym.vector.SyncVectorEnv(env_fns)

        # Populate specs
        obs_shape = self.envs.single_observation_space.shape
        obs_dtype = str(self.envs.single_observation_space.dtype)
        self.obs_spec = TensorSpec(shape=obs_shape, dtype=obs_dtype)
        
        if hasattr(self.envs.single_action_space, "n"):
            # Discrete
            self.act_spec = TensorSpec(shape=(), dtype="int64")
        else:
            # Continuous
            self.act_spec = TensorSpec(shape=self.envs.single_action_space.shape, dtype=str(self.envs.single_action_space.dtype))

        self.observation_space = self.envs.single_observation_space
        self.action_space = self.envs.single_action_space

    def reset(self, seed: Optional[int] = None) -> ObsBatch:
        """Reset all environments and return batched observations."""
        obs, info = self.envs.reset(seed=seed)
        return torch.from_numpy(obs).float()

    def step(self, actions: Any) -> 'StepResult':
        """Step all environments with a batch of actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
            
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
            obs=torch.from_numpy(obs).float(),
            reward=torch.from_numpy(reward).float(),
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
