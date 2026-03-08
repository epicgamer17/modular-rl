from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from torch import Tensor
import numpy as np

from replay_buffers.utils import discounted_cumulative_sums


class BaseTargetBuilder(ABC):
    """
    Abstract base class for calculating Reinforcement Learning targets.

    The TargetBuilder decouples the logic for computing targets (e.g., Bellman targets,
    MuZero unrolled targets, TD-lambda targets) from the Learner and the Losses.
    """

    @abstractmethod
    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Any,  # Union[Dict[str, Tensor], LearningOutput]
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        """
        Build target tensors for the loss calculation.

        Args:
            batch: A dictionary of tensors containing experience data from the replay buffer.
            predictions: A dictionary of tensors containing the network's current predictions.
            network: The neural network module (may be used for computing targets like target networks).

        Returns:
            A dictionary of target tensors (e.g., {"values": target_values, "policies": target_policies}).
        """
        pass  # pragma: no cover


class DQNTargetBuilder(BaseTargetBuilder):
    """
    Target builder for DQN and its variants (Double DQN, C51).
    Implements the Bellman equation logic.
    """

    def __init__(self, config: Any, device: torch.device):
        self.config = config
        self.device = device
        self.n_step = getattr(config, "n_step", 1)
        self.gamma = getattr(config, "discount_factor", 0.99)
        self.use_c51 = getattr(config, "atom_size", 1) > 1

        if self.use_c51:
            self.v_min = config.v_min
            self.v_max = config.v_max
            self.atom_size = config.atom_size
            self.support = torch.linspace(
                self.v_min, self.v_max, self.atom_size, device=self.device
            )

    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Any,
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        """
        Calculates DQN/C51 targets.
        """
        # Strictly enforce LearningOutput
        preds_dict = predictions._asdict()

        rewards = batch["rewards"].to(self.device).float()
        dones = batch["dones"].to(self.device).bool()
        terminated = batch.get("terminated", dones).to(self.device).bool()

        bootstrap_on_truncated = getattr(self.config, "bootstrap_on_truncated", False)
        terminal_mask = terminated if bootstrap_on_truncated else dones

        # Double DQN: Use online network (predictions) to select actions,
        # but target network to evaluate them.
        next_q_values = preds_dict.get("next_q_values")
        target_q_values = preds_dict.get("target_q_values")

        # If target_next_q_values is not provided in predictions,
        # it might need to be computed here if target_network is accessible.
        # However, following the refactor pattern, predictions should already contain it.

        batch_size = rewards.shape[0]
        discount = self.gamma**self.n_step

        if not self.use_c51:
            # Standard DQN Bellman
            # next_online_q_values: [B, num_actions]
            # target_next_q_values: [B, num_actions]

            next_actions = next_q_values.argmax(dim=-1)
            max_next_q = target_q_values[
                torch.arange(batch_size, device=self.device), next_actions
            ]

            target_q = rewards + discount * (~terminal_mask) * max_next_q
            return {"q_values": target_q}
        else:
            # C51 Distributional Bellman
            # predictions: {"next_q_logits": ..., "target_q_logits": ...}
            next_q_logits = preds_dict.get("next_q_logits")
            target_q_logits = preds_dict.get("target_q_logits")

            # 1. Select actions using online distribution
            online_next_probs = torch.softmax(next_q_logits, dim=-1)
            online_next_q = (online_next_probs * self.support).sum(dim=-1)
            next_actions = online_next_q.argmax(dim=-1)

            # 2. Get target distribution for those actions
            target_next_probs = torch.softmax(target_q_logits, dim=-1)
            chosen_target_next_probs = target_next_probs[
                torch.arange(batch_size, device=self.device), next_actions
            ]

            # 3. Project target distribution
            target_dist = self._project_target_distribution(
                rewards, terminal_mask, chosen_target_next_probs
            )
            return {"target_dist": target_dist}

    def _project_target_distribution(
        self,
        rewards: Tensor,
        terminal_mask: Tensor,
        next_probs: Tensor,
    ) -> Tensor:
        batch_size = rewards.shape[0]
        discount = self.gamma**self.n_step
        delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)

        tz = (
            rewards.view(-1, 1)
            + discount * (~terminal_mask).view(-1, 1) * self.support.view(1, -1)
        ).clamp(self.v_min, self.v_max)

        b = (tz - self.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # Handle indexing to avoid loops (vectorized scatter_add)
        projected = torch.zeros((batch_size, self.atom_size), device=self.device)

        # l and u are [B, atom_size]
        # next_probs is [B, atom_size]

        # We need to distribute next_probs[i, j] to projected[i, l[i, j]] and projected[i, u[i, j]]
        # based on distances.
        dist_l = u.float() - b
        dist_u = b - l.float()

        # Case where l == u (b is exact integer)
        mask_equal = l == u
        dist_l[mask_equal] = 1.0
        dist_u[mask_equal] = 0.0

        # projected[i].scatter_add_(0, l[i], next_probs[i] * dist_l[i])
        # We can use index_put_ or scatter_add with properly reshaped indices

        for j in range(self.atom_size):
            projected.scatter_add_(
                1, l[:, j : j + 1], (next_probs[:, j] * dist_l[:, j]).view(-1, 1)
            )
            projected.scatter_add_(
                1, u[:, j : j + 1], (next_probs[:, j] * dist_u[:, j]).view(-1, 1)
            )

        return projected


class PPOTargetBuilder(BaseTargetBuilder):
    """
    Target builder for PPO.
    Implements Generalized Advantage Estimation (GAE).
    """

    def __init__(self, config: Any, device: torch.device):
        self.config = config
        self.device = device
        self.gamma = getattr(config, "discount_factor", 0.99)
        self.gae_lambda = getattr(config, "gae_lambda", 0.95)

    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Any,
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        """
        Calculates GAE advantages and value targets.
        """
        # Strictly enforce LearningOutput
        preds_dict = predictions._asdict()

        # In PPO, we typically receive a batch that represents a full rollout or multiple rollouts.
        # If the batch already has advantages/returns, we might just return them,
        # but we follow the instruction to "Calculate".

        rewards = batch["rewards"].to(self.device).float()
        dones = batch["dones"].to(self.device).bool()
        values = preds_dict["values"].to(self.device).float()

        # We need the value of the 'next' state for the final transition in the buffer.
        # If the batch doesn't have it, we might need to assume it's zero or have it passed in.
        # Standard GAE implementation requires values[t+1].

        if values.shape[0] == rewards.shape[0] + 1:
            # We have the final bootstrap value
            v_all = values
        else:
            # We might need to assume 0 or handle it as best we can.
            # In our framework, PPO rollouts usually include the extra state or
            # the BufferProcessor handled it.
            v_all = torch.cat([values, torch.tensor([0.0], device=self.device)])

        rewards_np = rewards.cpu().numpy()
        values_np = v_all.cpu().numpy()
        dones_np = dones.cpu().numpy()

        deltas = rewards_np + self.gamma * values_np[1:] * (~dones_np) - values_np[:-1]

        advantages = discounted_cumulative_sums(deltas, self.gamma * self.gae_lambda)
        returns = discounted_cumulative_sums(rewards_np, self.gamma)

        return {
            "advantages": torch.from_numpy(advantages.copy()).to(self.device).float(),
            "returns": torch.from_numpy(returns.copy()).to(self.device).float(),
        }


class MuZeroTargetBuilder(BaseTargetBuilder):
    """
    Target builder for MuZero.
    Maps unrolled targets from the replay buffer to the learner format.
    """

    def __init__(self, config: Any, device: torch.device):
        self.config = config
        self.device = device

    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Any,
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        """
        Simply extracts pre-calculated targets from the batch.
        """
        targets = {}
        target_keys = [
            "values",
            "policies",
            "rewards",
            "action_mask",
            "obs_mask",
            "dones",
            "chance_codes",
        ]

        for k in target_keys:
            if k in batch:
                targets[k] = batch[k].to(self.device)

        # Handle EfficientZero consistency targets if present in batch
        if "consistency_targets" in batch:
            targets["consistency_targets"] = batch["consistency_targets"].to(
                self.device
            )

        return targets
