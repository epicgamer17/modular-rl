import torch
import numpy as np
from typing import Dict, Any, Generator, Tuple, Optional


class RolloutBuffer:
    """
    Rollout buffer for PPO.
    Stores transitions collected from vectorized environments.
    """

    def __init__(
        self, rollout_steps: int, num_envs: int, obs_dim: int, device: str = "cpu"
    ):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.device = torch.device(device)
        self.capacity = rollout_steps * num_envs

        self.reset()

    def reset(self):
        """Reset the buffer."""
        self.obs = torch.zeros(
            (self.rollout_steps, self.num_envs, self.obs_dim), device=self.device
        )
        self.actions = torch.zeros(
            (self.rollout_steps, self.num_envs), device=self.device
        )
        self.log_probs = torch.zeros(
            (self.rollout_steps, self.num_envs), device=self.device
        )
        self.rewards = torch.zeros(
            (self.rollout_steps, self.num_envs), device=self.device
        )
        self.terminateds = torch.zeros(
            (self.rollout_steps, self.num_envs), device=self.device
        )
        self.truncateds = torch.zeros(
            (self.rollout_steps, self.num_envs), device=self.device
        )
        self.values = torch.zeros(
            (self.rollout_steps, self.num_envs), device=self.device
        )
        self.policy_version = torch.zeros(
            (self.rollout_steps, self.num_envs), dtype=torch.long, device=self.device
        )

        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        policy_version: int = 0,
        env_idx: Optional[int] = None,
    ):
        """
        Add transition(s) to the buffer.

        Args:
            obs: [obs_dim] if env_idx is set, else [num_envs, obs_dim]
            action: scalar if env_idx is set, else [num_envs]
            reward: scalar if env_idx is set, else [num_envs]
            terminated: scalar if env_idx is set, else [num_envs]
            truncated: scalar if env_idx is set, else [num_envs]
            value: scalar if env_idx is set, else [num_envs]
            log_prob: scalar if env_idx is set, else [num_envs]
            policy_version: current policy version
            env_idx: index of the environment for single transition add
        """
        assert self.ptr < self.rollout_steps, "Buffer is full"

        # TODO: should env_idx always be required?
        if env_idx is not None:
            # Single transition add
            self.obs[self.ptr, env_idx].copy_(obs)
            self.actions[self.ptr, env_idx] = action
            self.rewards[self.ptr, env_idx] = reward
            self.terminateds[self.ptr, env_idx] = terminated
            self.truncateds[self.ptr, env_idx] = truncated
            self.values[self.ptr, env_idx] = value
            self.log_probs[self.ptr, env_idx] = log_prob
            self.policy_version[self.ptr, env_idx] = policy_version

            # Increment pointer only after all environments have been filled for this step
            if env_idx == self.num_envs - 1:
                self.ptr += 1
                if self.ptr == self.rollout_steps:
                    self.full = True
        else:
            # Batch add (all environments at once)
            self.obs[self.ptr].copy_(obs)
            self.actions[self.ptr].copy_(action)
            self.rewards[self.ptr].copy_(reward)
            self.terminateds[self.ptr].copy_(terminated)
            self.truncateds[self.ptr].copy_(truncated)
            self.values[self.ptr].copy_(value)
            self.log_probs[self.ptr].copy_(log_prob)
            self.policy_version[self.ptr].fill_(policy_version)

            self.ptr += 1
            if self.ptr == self.rollout_steps:
                self.full = True

    def get_all(self) -> "TransitionBatch":
        """
        Return all stored transitions as a TransitionBatch.
        Returns tensors in their structured shape [rollout_steps, num_envs, ...].
        """
        from core.batch import TransitionBatch
        
        # PPO doesn't explicitly store next_obs, we could reconstruct it from obs[1:] 
        # but for get_all() it's safer to just return zeros or None if not needed.
        # Since TransitionBatch expects a Tensor for next_obs in some contexts,
        # we'll provide a zero tensor of the same shape as obs for now.
        next_obs = torch.zeros_like(self.obs[: self.ptr])

        return TransitionBatch(
            obs=self.obs[: self.ptr],
            action=self.actions[: self.ptr],
            log_prob=self.log_probs[: self.ptr],
            reward=self.rewards[: self.ptr],
            next_obs=next_obs,
            done=(self.terminateds[: self.ptr].bool() | self.truncateds[: self.ptr].bool()).float(),
            terminated=self.terminateds[: self.ptr],
            truncated=self.truncateds[: self.ptr],
            value=self.values[: self.ptr],
            policy_version=self.policy_version[: self.ptr],
        )

    def iterate_minibatches(
        self, minibatch_size: int, extra_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Generator["TransitionBatch", None, None]:
        """
        Yield minibatches from the buffer as TransitionBatch objects.

        Args:
            minibatch_size: size of each minibatch
            extra_data: optional dictionary of additional tensors to include (e.g. advantages, returns)
        """
        from core.batch import TransitionBatch

        # Flatten all tensors up to ptr
        effective_size = self.ptr * self.num_envs
        b_obs = self.obs[: self.ptr].reshape((-1, self.obs_dim))
        b_log_probs = self.log_probs[: self.ptr].reshape(-1)
        b_actions = self.actions[: self.ptr].reshape(-1)
        b_values = self.values[: self.ptr].reshape(-1)
        b_terminateds = self.terminateds[: self.ptr].reshape(-1)
        b_truncateds = self.truncateds[: self.ptr].reshape(-1)
        b_policy_version = self.policy_version[: self.ptr].reshape(-1)
        b_rewards = self.rewards[: self.ptr].reshape(-1)

        # Handle extra data (e.g. advantages, returns)
        b_extra = {}
        if extra_data:
            for k, v in extra_data.items():
                b_extra[k] = v.reshape(-1)

        indices = np.arange(effective_size)
        np.random.shuffle(indices)

        for start in range(0, effective_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]

            # Reconstruct next_obs is not possible here as we are shuffling
            # For PPO minibatches, next_obs is generally not used.
            mb_next_obs = torch.zeros_like(b_obs[mb_indices])

            yield TransitionBatch(
                obs=b_obs[mb_indices],
                log_prob=b_log_probs[mb_indices],
                action=b_actions[mb_indices],
                value=b_values[mb_indices],
                reward=b_rewards[mb_indices],
                next_obs=mb_next_obs,
                done=(b_terminateds[mb_indices].bool() | b_truncateds[mb_indices].bool()).float(),
                terminated=b_terminateds[mb_indices],
                truncated=b_truncateds[mb_indices],
                policy_version=b_policy_version[mb_indices],
            )

    def clear(self):
        """Clear the buffer."""
        self.reset()

    def sample_query(
        self,
        batch_size: int,
        filters: Optional[Dict[str, Any]] = None,
        temporal_window: Optional[int] = None,
        contiguous: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compatibility method for LearnerRuntime.
        Returns a random minibatch.
        """
        if seed is not None:
            np.random.seed(seed)

        # We use the existing iterate_minibatches but just take one
        gen = self.iterate_minibatches(batch_size)
        try:
            return next(gen)
        except StopIteration:
            return {}

    def __len__(self):
        return self.ptr * self.num_envs if not self.full else self.capacity
