from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from torch import Tensor


class BaseTargetBuilder(ABC):
    """
    Abstract base class for Reinforcement Learning target calculation modules.
    """

    @abstractmethod
    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
        network: nn.Module,
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


class TemporalDifferenceBuilder(BaseTargetBuilder):
    """
    Standard TD target builder for Q-learning (Double DQN).
    Calculates: target = reward + gamma^n * (1 - done) * max_a' Q_target(s', a')
    """

    def __init__(
        self,
        target_network: nn.Module,
        gamma: float = 0.99,
        n_step: int = 1,
        bootstrap_on_truncated: bool = False,
    ):
        self.target_network = target_network
        self.gamma = gamma
        self.n_step = n_step
        self.bootstrap_on_truncated = bootstrap_on_truncated

    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        rewards = batch["rewards"].float()
        dones = batch["dones"].bool()
        terminated = batch.get("terminated", dones).bool()
        next_obs = batch.get("next_observations")
        next_masks = batch.get("next_legal_moves_masks")

        terminal_mask = terminated if self.bootstrap_on_truncated else dones
        batch_size = rewards.shape[0]
        discount = self.gamma**self.n_step

        with torch.inference_mode():
            # Double DQN: Use online network for action selection
            online_next_out = network.learner_inference({"observations": next_obs})
            next_q_values = online_next_out["q_values"]

            # Use target network for value estimation
            target_out = self.target_network.learner_inference(
                {"observations": next_obs}
            )
            target_q_values = target_out["q_values"]

        # Ensure shapes are [B, Actions]
        if next_q_values.dim() == 3:
            next_q_values = next_q_values.squeeze(1)
        if target_q_values.dim() == 3:
            target_q_values = target_q_values.squeeze(1)

        if next_masks is not None:
            next_q_values = next_q_values.masked_fill(~next_masks.bool(), -float("inf"))

        next_actions = next_q_values.argmax(dim=-1)
        max_next_q = target_q_values[
            torch.arange(batch_size, device=rewards.device), next_actions
        ]

        target_q = (rewards + discount * (~terminal_mask) * max_next_q).detach()
        return {
            "q_values": target_q,
            "actions": batch["actions"],
        }


class TDCategoricalProjectionBuilder(BaseTargetBuilder):
    """
    C51 distributional target builder.
    Performs L2-projection of the target distribution onto the support.
    """

    def __init__(
        self,
        target_network: nn.Module,
        v_min: float,
        v_max: float,
        atom_size: int,
        gamma: float = 0.99,
        n_step: int = 1,
        bootstrap_on_truncated: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.target_network = target_network
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.gamma = gamma
        self.n_step = n_step
        self.bootstrap_on_truncated = bootstrap_on_truncated
        self.device = device

        # Create support once
        self.support = torch.linspace(v_min, v_max, atom_size, device=device)

    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        rewards = batch["rewards"].float()
        dones = batch["dones"].bool()
        terminated = batch.get("terminated", dones).bool()
        next_obs = batch.get("next_observations")
        next_masks = batch.get("next_legal_moves_masks")

        terminal_mask = terminated if self.bootstrap_on_truncated else dones
        batch_size = rewards.shape[0]

        with torch.inference_mode():
            # Online next q logits for action selection
            online_next_out = network.learner_inference({"observations": next_obs})
            next_q_logits = online_next_out["q_logits"]

            # Target next distribution
            target_out = self.target_network.learner_inference(
                {"observations": next_obs}
            )
            target_q_logits = target_out["q_logits"]

        if next_q_logits.dim() == 4:
            next_q_logits = next_q_logits.squeeze(1)
        if target_q_logits.dim() == 4:
            target_q_logits = target_q_logits.squeeze(1)

        online_next_probs = torch.softmax(next_q_logits, dim=-1)
        online_next_q = (online_next_probs * self.support).sum(dim=-1)

        if next_masks is not None:
            online_next_q = online_next_q.masked_fill(~next_masks.bool(), -float("inf"))

        next_actions = online_next_q.argmax(dim=-1)

        target_next_probs = torch.softmax(target_q_logits, dim=-1)
        chosen_target_next_probs = target_next_probs[
            torch.arange(batch_size, device=rewards.device), next_actions
        ]

        q_logits = self._project_target_distribution(
            rewards, terminal_mask, chosen_target_next_probs
        ).detach()

        return {
            "q_logits": q_logits,
            "actions": batch["actions"],
        }

    def _project_target_distribution(
        self, rewards: Tensor, terminal_mask: Tensor, next_probs: Tensor
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

        projected = torch.zeros((batch_size, self.atom_size), device=rewards.device)
        dist_l = u.float() - b
        dist_u = b - l.float()

        mask_equal = l == u
        dist_l[mask_equal] = 1.0
        dist_u[mask_equal] = 0.0

        projected.scatter_add_(1, l, next_probs * dist_l)
        projected.scatter_add_(1, u, next_probs * dist_u)

        return projected


class LatentConsistencyBuilder(BaseTargetBuilder):
    """
    Build detached target embeddings for EfficientZero consistency loss.
    Takes None in __init__.
    """

    def __init__(self):
        pass

    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        # Use unroll_observations from buffer [B, T+1, C, H, W]
        # UniversalLearner already passed through the batch, so it's in batch
        real_obs = batch["unroll_observations"].float()
        batch_size, unroll_len = real_obs.shape[:2]
        flat_obs = real_obs.flatten(0, 1)

        initial_out = network.obs_inference(flat_obs)
        real_latents = initial_out.network_state.dynamics

        # Clone to promote from inference_mode tensors to normal autograd-tracked tensors
        real_latents = real_latents.clone()

        proj_targets = network.project(real_latents, grad=False)
        normalized_targets = torch.nn.functional.normalize(
            proj_targets, p=2.0, dim=-1, eps=1e-5
        )

        consistency_targets = normalized_targets.reshape(
            batch_size, unroll_len, -1
        ).detach()
        return {"consistency_targets": consistency_targets}


class TrajectoryGradientScaleBuilder(BaseTargetBuilder):
    """
    Builds gradient scaling tensors for BPTT unrolling.
    Ensures gradients are correctly weighted across the sequence.
    """

    def __init__(self, unroll_steps: int):
        self.unroll_steps = unroll_steps

    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        # Gradient Scales for MuZero sequence unrolling
        # Typically [1.0] for root, then [1/unroll_steps] for subsequent steps
        scales = [1.0] + [1.0 / self.unroll_steps] * self.unroll_steps
        # batch["rewards"] is used to get the device
        device = batch["rewards"].device
        scales_tensor = torch.tensor(scales, device=device).reshape(1, -1)
        return {"gradient_scales": scales_tensor}


class TargetBuilderPipeline(BaseTargetBuilder):
    """
    Target builder that delegates target building to a pipeline of builders.
    All builders sequentially update the target dictionary.
    """

    def __init__(self, builders: List[BaseTargetBuilder]):
        self.builders = builders

    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        targets = {}
        for builder in self.builders:
            targets.update(builder.build_targets(batch, predictions, network))
        return targets
