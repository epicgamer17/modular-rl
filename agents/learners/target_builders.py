from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from modules.world_models.inference_output import LearningOutput
from replay_buffers.utils import discounted_cumulative_sums


@dataclass
class TargetOutput:
    """
    Container for all possible target fields used by loss modules.
    Fields default to None if not computed by a specific builder.
    """

    q_values: Optional[Tensor] = None
    value_targets: Optional[Tensor] = None
    advantages: Optional[Tensor] = None
    old_log_probs: Optional[Tensor] = None
    policies: Optional[Tensor] = None
    rewards: Optional[Tensor] = None
    chance_codes: Optional[Tensor] = None
    consistency_targets: Optional[Tensor] = None
    target_dist: Optional[Tensor] = None
    values: Optional[Tensor] = None
    chance_values: Optional[Tensor] = None
    to_plays: Optional[Tensor] = None
    action_mask: Optional[Tensor] = None
    obs_mask: Optional[Tensor] = None
    dones: Optional[Tensor] = None
    actions: Optional[Tensor] = None
    returns: Optional[Tensor] = None
    target_policies: Optional[Tensor] = None


class BaseTargetBuilder(ABC):
    """
    Abstract base class for Reinforcement Learning target calculation modules.
    """

    @abstractmethod
    def build_targets(
        self, batch: Dict[str, Tensor], predictions: LearningOutput, network: nn.Module
    ) -> TargetOutput:
        """
        Build target tensors for the loss calculation.

        Args:
            batch: Dictionary of tensors from the replay buffer.
            predictions: Current network predictions (LearningOutput).
            network: The neural network module (may be used for target network calls).

        Returns:
            TargetOutput containing the computed targets.
        """
        pass  # pragma: no cover


class DQNTargetBuilder(BaseTargetBuilder):
    """
    Implements Bellman equation targets for DQN and C51.
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
        self, batch: Dict[str, Tensor], predictions: LearningOutput, network: nn.Module
    ) -> TargetOutput:
        rewards = batch["rewards"].to(self.device).float()
        dones = batch["dones"].to(self.device).bool()
        terminated = batch.get("terminated", dones).to(self.device).bool()
        next_masks = batch.get("next_legal_moves_masks")

        bootstrap_on_truncated = getattr(self.config, "bootstrap_on_truncated", False)
        terminal_mask = terminated if bootstrap_on_truncated else dones

        batch_size = rewards.shape[0]
        discount = self.gamma**self.n_step

        if not self.use_c51:
            next_q_values = predictions.next_q_values
            target_q_values = predictions.target_q_values

            if next_masks is not None:
                mask = next_masks.to(self.device).bool()
                next_q_values = next_q_values.masked_fill(~mask, -float("inf"))

            next_actions = next_q_values.argmax(dim=-1)
            max_next_q = target_q_values[
                torch.arange(batch_size, device=self.device), next_actions
            ]

            target_q = (rewards + discount * (~terminal_mask) * max_next_q).detach()
            return TargetOutput(
                q_values=target_q,
                values=target_q,
                actions=batch["actions"].to(self.device),
            )
        else:
            next_q_logits = predictions.next_q_logits
            target_q_logits = predictions.target_q_logits

            online_next_probs = torch.softmax(next_q_logits, dim=-1)
            online_next_q = (online_next_probs * self.support).sum(dim=-1)

            if next_masks is not None:
                mask = next_masks.to(self.device).bool()
                online_next_q = online_next_q.masked_fill(~mask, -float("inf"))

            next_actions = online_next_q.argmax(dim=-1)

            target_next_probs = torch.softmax(target_q_logits, dim=-1)
            chosen_target_next_probs = target_next_probs[
                torch.arange(batch_size, device=self.device), next_actions
            ]

            target_dist = self._project_target_distribution(
                rewards, terminal_mask, chosen_target_next_probs
            )
            target_dist = target_dist.detach()
            target_scalar = (target_dist * self.support).sum(dim=-1).detach()

            return TargetOutput(
                target_dist=target_dist,
                values=target_scalar,
                actions=batch["actions"].to(self.device),
            )

    def _project_target_distribution(
        self, rewards: Tensor, terminal_mask: Tensor, next_probs: Tensor
    ) -> Tensor:
        batch_size = rewards.shape[0]
        discount = self.gamma**self.n_step
        delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)

        tz = (
            rewards.view(-1, 1)
            + discount * (~terminal_mask.bool()).view(-1, 1) * self.support.view(1, -1)
        ).clamp(self.v_min, self.v_max)

        b = (tz - self.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        projected = torch.zeros((batch_size, self.atom_size), device=self.device)
        dist_l = u.float() - b
        dist_u = b - l.float()

        mask_equal = l == u
        dist_l[mask_equal] = 1.0
        dist_u[mask_equal] = 0.0

        for j in range(self.atom_size):
            projected.scatter_add_(
                1, l[:, j : j + 1], (next_probs[:, j] * dist_l[:, j]).view(-1, 1)
            )
            projected.scatter_add_(
                1, u[:, j : j + 1], (next_probs[:, j] * dist_u[:, j]).view(-1, 1)
            )

        return projected


class MuZeroTargetBuilder(BaseTargetBuilder):
    """
    Adapter logic to map pre-computed MCTS targets from batch to TargetOutput.
    """

    def build_targets(
        self, batch: Dict[str, Tensor], predictions: LearningOutput, network: nn.Module
    ) -> TargetOutput:
        return TargetOutput(
            values=batch.get("target_values"),
            policies=batch.get("target_policies"),
            rewards=batch.get("target_rewards"),
            action_mask=batch.get("action_mask"),
            obs_mask=batch.get("obs_mask"),
            dones=batch.get("dones"),
            chance_codes=batch.get("chance_codes"),
            consistency_targets=batch.get("consistency_targets"),
            actions=batch.get("actions"),
            target_policies=batch.get("target_policies"),
        )


class PPOTargetBuilder(BaseTargetBuilder):
    """
    Adapter for PPO targets. Assumes advantages and returns are pre-computed
    by the GAEProcessor in the replay buffer.
    """

    def __init__(self, config: Any, device: torch.device):
        self.config = config
        self.device = device

    def build_targets(
        self, batch: Dict[str, Tensor], predictions: LearningOutput, network: nn.Module
    ) -> TargetOutput:
        """
        Maps pre-computed advantages and returns from the batch.
        """
        return TargetOutput(
            advantages=batch["advantages"].to(self.device),
            value_targets=batch["returns"].to(self.device),
            returns=batch["returns"].to(self.device),
            old_log_probs=batch.get("log_probabilities").to(self.device),
            actions=batch["actions"].to(self.device),
        )


class TargetBuilderPipeline(BaseTargetBuilder):
    """
    Combines multiple target builders, merging their outputs.
    """

    def __init__(self, builders: List[BaseTargetBuilder]):
        self.builders = builders

    def build_targets(
        self, batch: Dict[str, Tensor], predictions: LearningOutput, network: nn.Module
    ) -> TargetOutput:
        combined = TargetOutput()
        for builder in self.builders:
            output = builder.build_targets(batch, predictions, network)
            for field_name, value in output.__dict__.items():
                if value is not None:
                    setattr(combined, field_name, value)
        return combined


class ImitationTargetBuilder(BaseTargetBuilder):
    """
    Target builder for imitation learning (Behavior Cloning).
    Extracts target policies from the batch.
    """

    def build_targets(
        self, batch: Dict[str, Tensor], predictions: LearningOutput, network: nn.Module
    ) -> TargetOutput:
        return TargetOutput(target_policies=batch.get("target_policies"))
