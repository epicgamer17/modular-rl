from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from replay_buffers.utils import discounted_cumulative_sums


class BaseTargetBuilder(ABC):
    """
    Abstract base class for Reinforcement Learning target calculation modules.
    """

    @abstractmethod
    def build_targets(
        self, batch: Dict[str, Tensor], predictions: Dict[str, Tensor], network: nn.Module
    ) -> Dict[str, Tensor]:
        """
        Build target tensors for the loss calculation.

        Args:
            batch: Dictionary of tensors from the replay buffer.
            predictions: Current network predictions (LearningOutput).
            network: The neural network module (may be used for target network calls).

        Returns:
            Dictionary containing the computed target tensors.
        """
        pass  # pragma: no cover


class DQNTargetBuilder(BaseTargetBuilder):
    """
    Implements Bellman equation targets for DQN and C51.
    """

    def __init__(
        self,
        device: torch.device,
        target_network: nn.Module,
        gamma: float = 0.99,
        n_step: int = 1,
        use_c51: bool = False,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        atom_size: int = 1,
        bootstrap_on_truncated: bool = False,
    ):
        self.device = device
        self.target_network = target_network
        self.gamma = gamma
        self.n_step = n_step
        self.use_c51 = use_c51
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.bootstrap_on_truncated = bootstrap_on_truncated

        if self.use_c51:
            assert (
                self.v_min is not None and self.v_max is not None
            ), "v_min and v_max must be provided for C51"
            self.support = torch.linspace(
                self.v_min, self.v_max, self.atom_size, device=self.device
            )

    def build_targets(
        self, batch: Dict[str, Tensor], predictions: Dict[str, Tensor], network: nn.Module
    ) -> Dict[str, Tensor]:
        rewards = batch["rewards"].to(self.device).float()
        dones = batch["dones"].to(self.device).bool()
        terminated = batch.get("terminated", dones).to(self.device).bool()
        next_masks = batch.get("next_legal_moves_masks")
        next_obs = batch.get("next_observations")  # Extract next_obs once

        terminal_mask = terminated if self.bootstrap_on_truncated else dones

        batch_size = rewards.shape[0]
        discount = self.gamma**self.n_step

        # Prepare next_obs if available
        if next_obs is not None:
            next_obs = next_obs.to(self.device).float()

        # TODO: MAKE SURE WE PREPROCESS OBS AND NEXT OBS HERE OR IN LEARNER
        if not self.use_c51:
            # 1. Get online network predictions for next state (Double DQN action selection)
            with torch.inference_mode():
                online_next_out = network.learner_inference({"observations": next_obs})
                next_q_values = online_next_out["q_values"]

            with torch.inference_mode():
                target_out = self.target_network.learner_inference(
                    {"observations": next_obs}
                )
                target_q_values = target_out["q_values"]

            # learner_inference returns [B, T+1, num_actions]; squeeze the T dim (always 1 here)
            if next_q_values.dim() == 3:
                assert (
                    next_q_values.shape[1] == 1
                ), f"Expected T=1 in next_q_values, got shape {next_q_values.shape}"
                next_q_values = next_q_values.squeeze(1)
            if target_q_values.dim() == 3:
                assert (
                    target_q_values.shape[1] == 1
                ), f"Expected T=1 in target_q_values, got shape {target_q_values.shape}"
                target_q_values = target_q_values.squeeze(1)

            if next_masks is not None:
                mask = next_masks.to(self.device).bool()
                next_q_values = next_q_values.masked_fill(~mask, -float("inf"))

            next_actions = next_q_values.argmax(dim=-1)
            max_next_q = target_q_values[
                torch.arange(batch_size, device=self.device), next_actions
            ]

            target_q = (rewards + discount * (~terminal_mask) * max_next_q).detach()
            return {
                "q_values": target_q,
                "actions": batch["actions"].to(self.device),
            }
        else:
            # 2. Get online network predictions for next state logits (C51 action selection)
            with torch.inference_mode():
                online_next_out = network.learner_inference({"observations": next_obs})
                next_q_logits = online_next_out["q_logits"]

                target_out = self.target_network.learner_inference(
                    {"observations": next_obs}
                )
                target_q_logits = target_out["q_logits"]

            # learner_inference returns [B, T+1, num_actions, atoms]; squeeze the T dim (always 1 here)
            if next_q_logits.dim() == 4:
                next_q_logits = next_q_logits.squeeze(1)
            if target_q_logits.dim() == 4:
                target_q_logits = target_q_logits.squeeze(1)

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

            q_logits = self._project_target_distribution(
                rewards, terminal_mask, chosen_target_next_probs
            ).detach()

            return {
                "q_logits": q_logits,
                "actions": batch["actions"].to(self.device),
            }

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

        projected.scatter_add_(1, l, next_probs * dist_l)
        projected.scatter_add_(1, u, next_probs * dist_u)

        return projected


class TargetBuilderPipeline(BaseTargetBuilder):
    """
    Target builder that delegates target building to a pipeline of builders.
    All builders sequentially update the target dictionary.
    """

    def __init__(self, builders: List[BaseTargetBuilder]):
        # Pipeline doesn't need its own config/device, it relies on the underlying builders
        self.builders = builders

    def build_targets(
        self, batch: Dict[str, Tensor], predictions: Dict[str, Tensor], network: nn.Module
    ) -> Dict[str, Tensor]:
        """
        Build targets by calling all builders in the pipeline.

        Args:
            batch: Dictionary of tensors from the replay buffer.
            predictions: Current network predictions.
            network: The neural network module.

        Returns:
            Dictionary containing the merged computed target tensors.
        """
        targets = {}
        for builder in self.builders:
            targets.update(builder.build_targets(batch, predictions, network))
        return targets


class MuZeroTargetBuilder(BaseTargetBuilder):
    """
    MuZero target builder that passes through buffer targets and adds consistency targets.
    """

    def __init__(self, config: Any, device: torch.device):
        self.config = config
        self.device = device

    def _prepare_consistency_targets(self, targets: Dict[str, Tensor], network: nn.Module) -> Tensor:
        """Build detached target embeddings for EfficientZero consistency loss."""
        # Use unroll_observations from buffer [B, T+1, C, H, W]
        real_obs = targets["unroll_observations"].to(self.device, dtype=torch.float32)
        batch_size, unroll_len = real_obs.shape[:2]
        flat_obs = real_obs.flatten(0, 1)

        # network.obs_inference is decorated with @torch.inference_mode()
        initial_out = network.obs_inference(flat_obs)
        real_latents = initial_out.network_state.dynamics

        # Clone to promote from inference_mode tensors (created by obs_inference's
        # @torch.inference_mode() decorator) to normal autograd-tracked tensors.
        # Without this, project() cannot save the tensor for backward.
        real_latents = real_latents.clone()

        proj_targets = network.project(real_latents, grad=False)
        normalized_targets = torch.nn.functional.normalize(proj_targets, p=2.0, dim=-1, eps=1e-5)
        return normalized_targets.reshape(batch_size, unroll_len, -1).detach()

    def _gradient_scales(self) -> torch.Tensor:
        unroll_steps = self.config.unroll_steps
        scales = [1.0] + [1.0 / unroll_steps] * unroll_steps
        return torch.tensor(scales, device=self.device).reshape(1, -1)

    def build_targets(
        self, batch: Dict[str, Tensor], predictions: Dict[str, Tensor], network: nn.Module
    ) -> Dict[str, Tensor]:
        """
        Pass through all targets already prepared by the MuZeroReplayBuffer.
        Compute consistency targets if enabled.
        """
        targets = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                targets[key] = value.to(self.device)

        # Consistency Targets (EfficientZero)
        if getattr(self.config, "consistency_loss_factor", 0) > 0:
            targets["consistency_targets"] = self._prepare_consistency_targets(targets, network)
        else:
            targets["consistency_targets"] = None

        # Gradient Scales for MuZero sequence unrolling
        targets["gradient_scales"] = self._gradient_scales()

        return targets
