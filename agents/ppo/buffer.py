import torch
import numpy as np
from typing import Dict, Any, Generator, Tuple, Optional


# TODO: this seems VERY PPO specific.. does this need to be its own class? We even hardcode GAE. The goal of this library is not to have Algorithm specific components (when possible, of course) but feature specific instead.
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

        self.advantages = torch.zeros(
            (self.rollout_steps, self.num_envs), device=self.device
        )
        self.returns = torch.zeros(
            (self.rollout_steps, self.num_envs), device=self.device
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
    ):
        """
        Add a batch of transitions to the buffer.

        Args:
            obs: [num_envs, obs_dim]
            action: [num_envs]
            reward: [num_envs]
            terminated: [num_envs]
            truncated: [num_envs]
            value: [num_envs]
            log_prob: [num_envs]
            policy_version: current policy version
        """
        assert self.ptr < self.rollout_steps, "Buffer is full"

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

    def compute_returns_advantages(
        self,
        next_value: torch.Tensor,
        next_terminated: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ):
        """
        Compute returns and advantages using GAE. Correct handles truncation (timeout masking)
        and bootstrapping.

        Args:
            next_value: [num_envs] value of the next state
            next_terminated: [num_envs] terminated flag (actual game over) of the next state
            gamma: discount factor
            gae_lambda: GAE parameter
        """
        last_gae_lam = 0
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                # For the last step in rollout, we use next_value if NOT terminated.
                # If it was truncated, we still bootstrap (next_terminated will be False).
                next_non_terminal = 1.0 - next_terminated.float()
                next_values = next_value
            else:
                # Within the rollout, we know exactly if it was terminated.
                # If it was truncated, we DO bootstrap (so non_terminal = 1).
                # If it was terminated, we DON'T bootstrap (so non_terminal = 0).
                next_non_terminal = 1.0 - self.terminateds[t + 1]
                next_values = self.values[t + 1]

            # delta_t = r_t + gamma * V(s_{t+1}) * nonterminal - V(s_t)
            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )

            # A_t = delta_t + gamma * lam * nonterminal * A_{t+1}
            self.advantages[t] = last_gae_lam = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            )

        self.returns = self.advantages + self.values

    def iterate_minibatches(
        self, minibatch_size: int
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Yield minibatches from the buffer.

        Args:
            minibatch_size: size of each minibatch
        """
        # Flatten all tensors
        b_obs = self.obs.reshape((-1, self.obs_dim))
        # Use log_probs (plural) to match iterate_minibatches yield
        b_log_probs = self.log_probs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)

        indices = np.arange(self.capacity)
        np.random.shuffle(indices)

        for start in range(0, self.capacity, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]

            yield {
                "obs": b_obs[mb_indices],
                "log_prob": b_log_probs[mb_indices],
                "action": b_actions[mb_indices],
                "advantages": b_advantages[mb_indices],
                "returns": b_returns[mb_indices],
                "values": b_values[mb_indices],
            }

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
