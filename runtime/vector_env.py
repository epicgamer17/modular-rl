import gymnasium as gym
from typing import Callable, List, Optional, Any
import numpy as np


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
class VectorEnv:
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

        env_fns = [
            make_env(env_id, seed + i, i, capture_video, run_name)
            for i in range(num_envs)
        ]

        if async_mode:
            self.envs = gym.vector.AsyncVectorEnv(env_fns)
        else:
            self.envs = gym.vector.SyncVectorEnv(env_fns)

        self.observation_space = self.envs.single_observation_space
        self.action_space = self.envs.single_action_space

    def reset(self, seed: Optional[int] = None):
        """Reset all environments."""
        return self.envs.reset(seed=seed)

    def step(self, actions: np.ndarray):
        """Step all environments with a batch of actions."""
        return self.envs.step(actions)

    def close(self):
        """Close all environments."""
        self.envs.close()

    @property
    def single_observation_space(self):
        return self.envs.single_observation_space

    @property
    def single_action_space(self):
        return self.envs.single_action_space
