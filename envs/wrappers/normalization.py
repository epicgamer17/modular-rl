import numpy as np
import gymnasium as gym
from typing import Any, Tuple, Dict, Union


class RunningMeanStd:
    """
    Tracks running mean and variance for observation normalization.
    Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update mean and variance with a batch of observations."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        """Update mean and variance from batch moments."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class NormalizeObservation:
    """
    Gymnasium wrapper to normalize observations using running mean and standard deviation.
    Supports both single and vectorized environments.
    """

    def __init__(self, env: Any, epsilon: float = 1e-8):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = isinstance(env, gym.vector.VectorEnv) or getattr(env, "is_vector_env", self.num_envs > 1) or hasattr(env, "num_envs")
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.is_vector_env:
            self.obs_rms.update(obs)
        else:
            self.obs_rms.update(obs[None])
        return self._normalize(obs), reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        if self.is_vector_env:
            self.obs_rms.update(obs)
        else:
            self.obs_rms.update(obs[None])
        return self._normalize(obs), info

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)
