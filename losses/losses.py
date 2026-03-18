import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
from torch import Tensor
from modules.world_models.inference_output import InferenceOutput
from utils.telemetry import append_metric
from losses.representations import (
    BaseRepresentation,
    get_representation,
    ScalarRepresentation,
    TwoHotRepresentation,
    CategoricalRepresentation,
    ExponentialBucketsRepresentation,
    ClassificationRepresentation,
)
from losses.priority_computers import (
    PriorityComputer,
    SpecificLossPriority,
    ErrorPriority,
)


class LossModule(ABC):
    """
    Unified base class for all loss modules.
    Works for both single-step (DQN, C51) and sequence (MuZero) losses.
    """

    def __init__(
        self,
        config,
        device,
        representation: Optional[BaseRepresentation] = None,
        optimizer_name: str = "default",
    ):
        self.config = config
        self.device = device
        self.optimizer_name = optimizer_name
        self.name = self.__class__.__name__
        self.representation = representation or self._build_default_representation()

    def _build_default_representation(self) -> BaseRepresentation:
        """Fallback to build representation directly from config if not provided explicitly."""
        support_range = getattr(self.config, "support_range", None)
        vmin = getattr(self.config, "v_min", None)
        vmax = getattr(self.config, "v_max", None)
        bins = getattr(self.config, "atom_size", None)
        mode = getattr(self.config, "representation_mode", "linear")
        num_classes = getattr(self.config, "num_classes", None)

        if vmin is not None and vmax is not None and bins is not None and bins > 1:
            # Force categorical mode if this is a C51Loss
            if self.__class__.__name__ == "C51Loss" and mode == "linear":
                mode = "categorical"
            return get_representation(vmin=vmin, vmax=vmax, bins=bins, mode=mode)
        if support_range is not None:
            return get_representation(support_range=support_range)
        if num_classes is not None and num_classes > 1:
            return ClassificationRepresentation(num_classes)
        return ScalarRepresentation()

    @property
    @abstractmethod
    def required_predictions(self) -> set[str]:
        """Set of keys required in the predictions dict."""
        pass  # pragma: no cover

    @property
    @abstractmethod
    def required_targets(self) -> set[str]:
        """Set of keys required in the targets dict."""
        pass  # pragma: no cover

    def should_compute(self, k: int, context: dict) -> bool:
        """Determine if this loss should be computed at step k."""
        return True

    def get_mask(self, context: dict) -> torch.Tensor:
        """Get the full mask to apply for this loss across all steps."""
        if "has_valid_obs_mask" in context:
            return context["has_valid_obs_mask"]
        # Fallback to ones if no mask provided
        return torch.ones(self.config.minibatch_size, 1, device=self.device)

    @abstractmethod
    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """
        Compute elementwise loss for all unroll steps.

        Returns:
            elementwise_tensor of shape (B, T) or (B, T, atoms)
        """
        pass  # pragma: no cover


# ============================================================================
# OLD DQN-STYLE LOSSES (Updated to work with unified interface)
# ============================================================================


class StandardDQNLoss(LossModule):
    def __init__(
        self,
        config,
        device,
        representation: Optional[BaseRepresentation] = None,
        action_selector: Optional[object] = None,
        optimizer_name: str = "default",
    ):
        super().__init__(config, device, representation, optimizer_name=optimizer_name)
        self.action_selector = action_selector

    @property
    def required_predictions(self) -> set[str]:
        return {"q_values"}

    @property
    def required_targets(self) -> set[str]:
        return {"values", "actions"}

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict
    ) -> torch.Tensor:
        q_values = predictions["q_values"]
        actions = targets["actions"].to(self.device).long()

        # Handle both [B, A] and [B, T, A]
        if q_values.ndim == 3:
            # Sequence case: [B, T, A]
            selected_q = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        else:
            # Single-step case: [B, A]
            selected_q = q_values[
                torch.arange(q_values.shape[0], device=self.device), actions
            ]

        targets_val = self.representation.format_target(targets)
        # Return elementwise loss (B, T) or (B,)
        return self.config.loss_function(selected_q, targets_val, reduction="none")


class C51Loss(LossModule):
    def __init__(
        self,
        config,
        device,
        representation: Optional[BaseRepresentation] = None,
        action_selector: Optional[object] = None,
        optimizer_name: str = "default",
    ):
        assert representation is not None, "C51Loss requires a representation."

        super().__init__(config, device, representation, optimizer_name=optimizer_name)
        self.action_selector = action_selector

    @property
    def required_predictions(self) -> set[str]:
        return {"q_logits"}

    @property
    def required_targets(self) -> set[str]:
        return {"actions", "next_q_logits", "next_actions", "rewards", "dones"}

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict
    ) -> torch.Tensor:
        online_q_logits = predictions["q_logits"]
        actions = targets["actions"].to(self.device).long()

        # Use representation to project target distribution (handles sequence [B, T])
        target_dist = self.representation.format_target(targets).detach()

        # Extract chosen logits: [B, T, Atoms] or [B, Atoms]
        if online_q_logits.ndim == 4:
            # [B, T, Actions, Atoms]
            actions_unsqueezed = (
                actions.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, -1, online_q_logits.shape[-1])
            )
            chosen_logits = online_q_logits.gather(-2, actions_unsqueezed).squeeze(-2)
        else:
            # [B, Actions, Atoms]
            chosen_logits = online_q_logits[
                torch.arange(online_q_logits.shape[0], device=self.device), actions
            ]

        log_probs = F.log_softmax(chosen_logits, dim=-1)
        return -(target_dist * log_probs).sum(dim=-1)


# ============================================================================
# NEW MUZERO-STYLE LOSSES (Updated to work with unified interface)
# ============================================================================


class ValueLoss(LossModule):
    """Value prediction loss module."""

    def __init__(
        self,
        config,
        device,
        representation: Optional[BaseRepresentation] = None,
        optimizer_name: str = "default",
    ):
        super().__init__(config, device, representation, optimizer_name=optimizer_name)

    @property
    def required_predictions(self) -> set[str]:
        deps = {"values"}
        if getattr(self.config, "latent_viz_interval", 0) > 0:
            deps.add("latents")
        return deps

    @property
    def required_targets(self) -> set[str]:
        deps = {"values"}
        if getattr(self.config, "latent_viz_interval", 0) > 0:
            deps.add("actions")
        return deps

    def should_compute(self, k: int, context: dict) -> bool:
        return True  # Compute at all steps

    def get_mask(self, context: dict) -> torch.Tensor:
        # Value loss is computed for all steps in the same game, even after terminal
        return context["is_same_game"]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        values = predictions["values"]
        target_rep = self.representation.format_target(targets).detach()

        # For Regression (Scalar) heads, squeeze (B, T, 1) -> (B, T)
        predicted_values = values
        if isinstance(self.representation, ScalarRepresentation):
            predicted_values = values.squeeze(-1)

        assert (
            predicted_values.shape == target_rep.shape
        ), f"Shape mismatch in ValueLoss: {predicted_values.shape} vs {target_rep.shape}"

        # Value Loss calculation
        value_loss = self.config.value_loss_function(
            predicted_values, target_rep, reduction="none"
        )

        # If the loss is elementwise (B, T, atoms), sum it up
        if value_loss.ndim == target_rep.ndim + 1:
            value_loss = value_loss.sum(dim=-1)

        value_loss = self.config.value_loss_factor * value_loss

        # Latent Visualization (Root k=0)
        viz_interval = getattr(self.config, "latent_viz_interval", 0)
        if viz_interval > 0 and context.get("training_step", 0) % viz_interval == 0:
            from utils.telemetry import add_latent_visualization_metric

            s0 = predictions.get("latents")
            if s0 is not None:
                # If sequence, take root
                if s0.ndim == 3:
                    s0 = s0[:, 0]
                actions = targets.get("actions")
                if actions is not None:
                    if actions.ndim == 2:
                        actions = actions[:, 0]
                    metrics = context.setdefault("metrics", {})
                    add_latent_visualization_metric(
                        metrics,
                        "latent_root",
                        s0.detach().cpu(),
                        labels=actions.detach().cpu(),
                        method=getattr(self.config, "latent_viz_method", "pca"),
                    )

        return value_loss



class PolicyLoss(LossModule):
    """Policy prediction loss module."""

    def __init__(
        self,
        config,
        device,
        representation: Optional[BaseRepresentation] = None,
        optimizer_name: str = "default",
    ):
        super().__init__(config, device, representation, optimizer_name=optimizer_name)

    @property
    def required_predictions(self) -> set[str]:
        return {"policies"}

    @property
    def required_targets(self) -> set[str]:
        return {"policies"}

    def should_compute(self, k: int, context: dict) -> bool:
        return True  # Compute at all steps

    def get_mask(self, context: dict) -> torch.Tensor:
        # IMPORTANT: Policy Loss uses Policy Mask (excludes terminal)
        return context["has_valid_action_mask"]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        policies = predictions["policies"]
        target_policies = targets["policies"]

        if self.config.policy_loss_function == F.kl_div:
            log_probs = F.log_softmax(policies, dim=-1)
            policy_loss = self.config.policy_loss_function(log_probs, target_policies, reduction="none")
            if policy_loss.ndim > target_policies.ndim - 1:
                policy_loss = policy_loss.sum(dim=-1)
        else:
            policy_loss = self.config.policy_loss_function(policies, target_policies, reduction="none")

        return policy_loss


class RewardLoss(LossModule):
    """Reward prediction loss module."""

    def __init__(
        self,
        config,
        device,
        representation: Optional[BaseRepresentation] = None,
        optimizer_name: str = "default",
    ):
        super().__init__(config, device, representation, optimizer_name=optimizer_name)

    @property
    def required_predictions(self) -> set[str]:
        return {"rewards"}

    @property
    def required_targets(self) -> set[str]:
        return {"rewards"}

    def should_compute(self, k: int, context: dict) -> bool:
        return k > 0  # Only for k > 0

    def get_mask(self, context: dict) -> torch.Tensor:
        # Reward loss is computed for all steps in the same game, even after terminal
        return context["is_same_game"]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        rewards = predictions["rewards"]
        # targets["rewards"]: [B, K]
        target_rep = self.representation.format_target(
            targets, target_key="rewards"
        ).detach()

        predicted_rewards = rewards
        if isinstance(self.representation, ScalarRepresentation):
            predicted_rewards = rewards.squeeze(-1)

        assert (
            predicted_rewards.shape == target_rep.shape
        ), f"Shape mismatch in RewardLoss: {predicted_rewards.shape} vs {target_rep.shape}"

        # Reward Loss calculation: [B, K]
        reward_loss_seq = self.config.reward_loss_function(
            predicted_rewards, target_rep, reduction="none"
        )

        # If the loss is elementwise (B, K, atoms), sum it up
        if reward_loss_seq.ndim == target_rep.ndim + 1:
            reward_loss_seq = reward_loss_seq.sum(dim=-1)

        # Pad to [B, T] where T = K+1
        # We assume predictions["values"] or context["is_same_game"] reflects T
        T = context["is_same_game"].shape[1]
        full_loss = torch.zeros(
            (reward_loss_seq.shape[0], T), device=reward_loss_seq.device
        )
        full_loss[:, 1:] = reward_loss_seq

        return self.config.reward_loss_factor * full_loss


class ToPlayLoss(LossModule):
    """To-play (turn indicator) prediction loss module."""

    def __init__(
        self,
        config,
        device,
        representation: Optional[BaseRepresentation] = None,
        optimizer_name: str = "default",
    ):
        super().__init__(config, device, representation, optimizer_name=optimizer_name)

    @property
    def required_predictions(self) -> set[str]:
        return {"to_plays"}

    @property
    def required_targets(self) -> set[str]:
        return {"to_plays"}

    def should_compute(self, k: int, context: dict) -> bool:
        # Only compute for multi-player games and k > 0
        return k > 0 and self.config.game.num_players != 1

    def get_mask(self, context: dict) -> torch.Tensor:
        # To-play exists for the terminal state too
        return context["has_valid_obs_mask"]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        if self.config.game.num_players == 1:
            # Should not happen if should_compute is checked, but return 0 to be safe
            return torch.zeros_like(targets["to_plays"])

        to_plays = predictions["to_plays"]
        target_rep = self.representation.format_target(targets).detach()

        # To-Play Loss: (B, T)
        to_play_loss = self.config.to_play_loss_factor * self.config.to_play_loss_function(
            to_plays, target_rep, reduction="none"
        )

        # Zero out k=0 if sequence has it (needs k > 0)
        if to_play_loss.ndim == 2:
            to_play_loss[:, 0] = 0.0

        # Logging
        metrics = context.setdefault("metrics", {})
        from utils.telemetry import append_metric

        probs = torch.softmax(to_plays, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        append_metric(metrics, "to_play_entropy", entropy.mean().item())

        return to_play_loss


class RelativeToPlayLoss(LossModule):
    """
    To-play loss for relative turn shifts (ΔP).
    Calculates ΔP targets from the sequence of absolute player indices:
    ΔP_k = (P_k - P_{k-1}) mod num_players.
    """

    def __init__(self, config, device, optimizer_name: str = "default"):
        super().__init__(config, device, optimizer_name=optimizer_name)

    @property
    def required_predictions(self) -> set[str]:
        return {"to_plays"}

    @property
    def required_targets(self) -> set[str]:
        # Needs to_plays to calculate delta
        return {"to_plays"}

    def get_mask(self, context: dict) -> torch.Tensor:
        return context["has_valid_obs_mask"]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        if self.config.game.num_players <= 1:
            return torch.zeros_like(targets["to_plays"])

        delta_p_logits = predictions["to_plays"]

        # Calculate target ΔP_k = (P_k - P_{k-1}) mod N
        p_seq = targets["to_plays"]
        num_players = self.config.game.num_players

        # p_seq: [B, T]. We need ΔP_k for k=1..T-1.
        # k=0 is undefined/zeroed.
        p_k = p_seq[:, 1:]
        p_prev = p_seq[:, :-1]
        target_delta_p = (p_k - p_prev) % num_players

        # Project ΔP_k index -> target distribution
        target_rep = self.representation.to_representation(target_delta_p).detach()

        # Loss calculation: [B, T-1]
        loss = self.config.to_play_loss_factor * self.config.to_play_loss_function(
            delta_p_logits[:, 1:], target_rep, reduction="none"
        )

        # Padding to [B, T]
        full_loss = torch.zeros_like(p_seq)
        full_loss[:, 1:] = loss

        return full_loss


class ConsistencyLoss(LossModule):
    """Consistency loss module (EfficientZero style)."""

    def __init__(
        self,
        config,
        device,
        agent_network,
        representation: Optional[BaseRepresentation] = None,
        optimizer_name: str = "default",
    ):
        super().__init__(
            config, device, representation=representation, optimizer_name=optimizer_name
        )
        self.agent_network = agent_network

    @property
    def required_predictions(self) -> set[str]:
        return {"latents"}

    @property
    def required_targets(self) -> set[str]:
        return {"consistency_targets"}

    def get_mask(self, context: dict) -> torch.Tensor:
        # Consistency valid if policy is valid (step is not terminal)
        return context["has_valid_action_mask"]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        latent_states = predictions["latents"]
        target_features = targets["consistency_targets"]

        # Flatten [B, T, D] to [B*T, D] for projection
        original_shape = latent_states.shape
        if latent_states.ndim == 3:
            B, T, D = latent_states.shape
            latent_states = latent_states.reshape(-1, D)
            target_features = target_features.reshape(-1, target_features.shape[-1])

        # Project, then predict
        proj_preds = self.agent_network.project(latent_states, grad=True)
        f2 = F.normalize(proj_preds, p=2.0, dim=-1, eps=1e-5)

        # Cosine similarity
        consistency = -(target_features * f2).sum(dim=1)
        consistency_loss = self.config.consistency_loss_factor * consistency

        # Reshape back to [B, T]
        if len(original_shape) == 3:
            consistency_loss = consistency_loss.view(
                original_shape[0], original_shape[1]
            )
            # Consistency usually only for k > 0
            consistency_loss[:, 0] = 0.0

        return consistency_loss


class ChanceQLoss(LossModule):
    """Q-value loss for chance nodes (stochastic MuZero)."""

    def __init__(
        self,
        config,
        device,
        representation: Optional[BaseRepresentation] = None,
        optimizer_name: str = "default",
    ):
        super().__init__(
            config, device, representation=representation, optimizer_name=optimizer_name
        )

    @property
    def required_predictions(self) -> set[str]:
        return {"chance_values"}

    @property
    def required_targets(self) -> set[str]:
        # Uses target_values_next which is targets["values"][:, k]
        return {"values"}

    def get_mask(self, context: dict) -> torch.Tensor:
        # Chance Q target is value from next step. Compute if both are in same game.
        return context["is_same_game"]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        if not self.config.stochastic:
            return torch.zeros_like(targets["values"])

        chance_values = predictions["chance_values"]
        # Target derived from replay values at k+1
        # targets["values"]: [B, T] (where T = K+1)
        # target_v_next: [B, K] (values at k=1..K)
        target_v_next = targets["values"][:, 1:]

        target_rep = self.representation.to_representation(target_v_next).detach()
        # chance_values already has size K (unroll steps)
        predicted_chance_values = chance_values

        if isinstance(self.representation, ScalarRepresentation):
            predicted_chance_values = predicted_chance_values.squeeze(-1)

        q_loss_seq = self.config.value_loss_function(
            predicted_chance_values, target_rep, reduction="none"
        )
        if q_loss_seq.ndim == target_rep.ndim + 1:
            q_loss_seq = q_loss_seq.sum(dim=-1)

        q_loss = self.config.value_loss_factor * q_loss_seq

        # Pad to [B, T] by prepending zero for k=0
        full_loss = torch.zeros_like(targets["values"])
        full_loss[:, 1:] = q_loss

        # Entropy logging (approximate)
        if self.config.support_range is not None:
            metrics = context.setdefault("metrics", {})
            probs = torch.softmax(predicted_chance_values, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            append_metric(metrics, "chance_q_entropy", entropy.mean().item())

        return full_loss


class SigmaLoss(LossModule):
    """Sigma (chance code prediction) loss for stochastic MuZero."""

    def __init__(
        self,
        config,
        device,
        representation: Optional[BaseRepresentation] = None,
        optimizer_name: str = "default",
    ):
        super().__init__(config, device, representation, optimizer_name=optimizer_name)

    @property
    def required_predictions(self) -> set[str]:
        return {"chance_logits"}

    @property
    def required_targets(self) -> set[str]:
        return {"chance_codes"}

    def should_compute(self, k: int, context: dict) -> bool:
        return self.config.stochastic and k > 0

    def get_mask(self, context: dict) -> torch.Tensor:
        return context["has_valid_action_mask"]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        if not self.config.stochastic:
            return torch.zeros_like(
                targets["chance_codes"], dtype=torch.float32
            ).squeeze(-1)

        chance_logits = predictions["chance_logits"]
        # target_codes: [B, T]. k=0 is invalid/padding.
        # chance_logits has size K (unroll steps k=1..K)
        target_codes_seq = targets["chance_codes"][:, 1:].squeeze(-1).long()
        
        if self.config.sigma_loss == F.cross_entropy:
            # cross_entropy(input[B, C, K], target[B, K])
            loss_seq = F.cross_entropy(
                chance_logits.transpose(1, 2), target_codes_seq, reduction="none"
            )
        else:
            one_hot = F.one_hot(
                target_codes_seq, num_classes=chance_logits.shape[-1]
            ).float()
            loss_seq = self.config.sigma_loss(
                chance_logits, one_hot.detach(), reduction="none"
            )
            if loss_seq.ndim == target_codes_seq.ndim + 1:
                loss_seq = loss_seq.sum(dim=-1)

        # Padding to [B, T]
        B, T = targets["chance_codes"].shape[:2]
        full_loss = torch.zeros((B, T), device=chance_logits.device)
        full_loss[:, 1:] = loss_seq

        # Logging
        metrics = context.setdefault("metrics", {})
        entropy = (
            -torch.softmax(chance_logits, dim=-1)
            * torch.log_softmax(chance_logits, dim=-1)
        ).sum(dim=-1)
        append_metric(metrics, "chance_entropy", entropy.mean().item())

        return full_loss


class VQVAECommitmentLoss(LossModule):
    """VQ-VAE commitment cost for encoder (stochastic MuZero)."""

    def __init__(
        self,
        config,
        device,
        representation: Optional[BaseRepresentation] = None,
        optimizer_name: str = "default",
    ):
        super().__init__(config, device, representation, optimizer_name=optimizer_name)

    @property
    def required_predictions(self) -> set[str]:
        return {"chance_encoder_embeddings"}

    @property
    def required_targets(self) -> set[str]:
        return {"chance_codes"}

    def should_compute(self, k: int, context: dict) -> bool:
        return (
            self.config.stochastic and k > 0 and not self.config.use_true_chance_codes
        )

    def get_mask(self, context: dict) -> torch.Tensor:
        # no chance nodes from terminal -> absorbing
        return context["has_valid_action_mask"]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        if not self.config.stochastic or self.config.use_true_chance_codes:
            return torch.zeros_like(
                targets["chance_codes"], dtype=torch.float32
            ).squeeze(-1)

        chance_enc_embeddings = predictions["chance_encoder_embeddings"]
        # chance_enc_embeddings has size K
        target_codes_seq = targets["chance_codes"][:, 1:].squeeze(-1).long()
        one_hot = F.one_hot(target_codes_seq, num_classes=chance_enc_embeddings.shape[-1]).float()

        diff = chance_enc_embeddings - one_hot.detach()
        vqvae_loss_seq = self.config.vqvae_commitment_cost_factor * torch.sum(diff.pow(2), dim=-1)
        
        # Padding to [B, T]
        B, T = targets["chance_codes"].shape[:2]
        full_loss = torch.zeros((B, T), device=vqvae_loss_seq.device)
        full_loss[:, 1:] = vqvae_loss_seq
        return full_loss


class PPOPolicyLoss(LossModule):
    def __init__(
        self,
        config,
        device,
        clip_param: float,
        entropy_coefficient: float,
        policy_strategy: Optional[object] = None,
        representation: Optional[BaseRepresentation] = None,
        optimizer_name: str = "default",
    ):
        super().__init__(
            config, device, representation=representation, optimizer_name=optimizer_name
        )
        self.clip_param = clip_param
        self.entropy_coefficient = entropy_coefficient
        self.policy_strategy = policy_strategy

    @property
    def required_predictions(self) -> set[str]:
        return {"policies"}

    @property
    def required_targets(self) -> set[str]:
        return {"actions", "old_log_probs", "advantages"}

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict
    ) -> torch.Tensor:
        policy_logits = predictions["policies"]
        actions = targets["actions"]
        old_log_probs = targets["old_log_probs"]
        advantages = targets["advantages"]

        if self.policy_strategy is not None:
            dist = self.policy_strategy.get_distribution(policy_logits)
        else:
            dist = torch.distributions.Categorical(logits=policy_logits)
        log_probs = dist.log_prob(actions)

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages

        entropy = dist.entropy()
        loss = -torch.min(surr1, surr2) - self.entropy_coefficient * entropy

        with torch.no_grad():
            metrics = context.setdefault("metrics", {})
            append_metric(metrics, "ppo_entropy", entropy.mean().item())

        return loss


class PPOValueLoss(LossModule):
    def __init__(
        self,
        config,
        device,
        critic_coefficient: float,
        atom_size: int = 1,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        value_strategy: Optional[object] = None,
        representation: Optional[BaseRepresentation] = None,
        optimizer_name: str = "default",
    ):
        super().__init__(
            config, device, representation=representation, optimizer_name=optimizer_name
        )
        self.critic_coefficient = critic_coefficient
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.value_strategy = value_strategy

    @property
    def required_predictions(self) -> set[str]:
        return {"values"}

    @property
    def required_targets(self) -> set[str]:
        return {"returns"}

    def _to_scalar_values(self, value_logits: torch.Tensor) -> torch.Tensor:
        if self.value_strategy is not None:
            return self.value_strategy.to_expected_value(value_logits)

        return self.representation.to_scalar(value_logits)

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict
    ) -> torch.Tensor:
        value_logits = predictions["values"]
        returns = targets["returns"]
        values = self._to_scalar_values(value_logits)
        return self.critic_coefficient * ((returns - values) ** 2)


class ImitationLoss(LossModule):
    def __init__(
        self,
        config,
        device,
        num_actions: int,
        optimizer_name: str = "default",
        representation: Optional[BaseRepresentation] = None,
    ):
        representation = representation or get_representation(num_classes=num_actions)
        super().__init__(
            config, device, representation=representation, optimizer_name=optimizer_name
        )
        self.num_actions = num_actions
        self.loss_function = getattr(
            config, "loss_function", torch.nn.CrossEntropyLoss(reduction="none")
        )

    @property
    def required_predictions(self) -> set[str]:
        return {"policies"}

    @property
    def required_targets(self) -> set[str]:
        return {"target_policies"}

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict
    ) -> torch.Tensor:
        policy_logits = predictions["policies"]
        target_policies = targets["target_policies"]

        target_rep = self.representation.to_representation(target_policies).detach()

        # Record policy mean for monitoring
        metrics = context.setdefault("metrics", {})
        append_metric(metrics, "sl_policy", torch.softmax(policy_logits, dim=-1).mean(dim=0).detach().cpu())

        loss = self.loss_function(policy_logits, target_rep)
        if loss.dim() > target_policies.dim():
            loss = loss.sum(dim=-1)
        return loss


# ============================================================================
# UNIFIED LOSS PIPELINE
# ============================================================================


class LossPipeline:
    """
    Unified pipeline that handles both single-step (DQN) and sequence (MuZero) losses.
    Validated at initialization to ensure all required keys are present.
    """

    def __init__(
        self,
        modules: List[LossModule],
        priority_computer: Optional[PriorityComputer] = None,
    ):
        super().__init__()
        self.loss_modules = nn.ModuleList(modules)
        self.priority_computer = priority_computer

    def validate_dependencies(
        self, network_output_keys: set[str], target_keys: set[str]
    ) -> None:
        """
        Verify that the provided keys satisfy all module requirements.
        Raises ValueError with detailed error message on failure.
        """
        for module in self.loss_modules:
            missing_preds = module.required_predictions - network_output_keys
            missing_targets = module.required_targets - target_keys

            if missing_preds:
                raise ValueError(
                    f"Module {module.name} missing required predictions: {missing_preds}. "
                    f"Available: {network_output_keys}"
                )
            if missing_targets:
                raise ValueError(
                    f"Module {module.name} missing required targets: {missing_targets}. "
                    f"Available: {target_keys}"
                )

    def run(
        self,
        predictions: dict,
        targets: dict,
        context: dict = {},
        weights: Optional[torch.Tensor] = None,
        gradient_scales: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], torch.Tensor]:
        """
        Run the loss pipeline across all unroll steps.

        Args:
            predictions: Dict of tensors with shape (B, K+1, ...) or (B, ...)
            targets: Dict of tensors with shape (B, K+1, ...) or (B, ...)
            context: Additional context (masks, etc.)
            weights: PER weights of shape (B,)
            gradient_scales: Gradient scales of shape (1, K+1)

        Returns:
            total_loss_dict: Dictionary mapping optimizer names to scalar losses
            loss_dict: Dictionary of accumulated losses for logging
            priorities: Priority tensor of shape (B,) for PER
        """
        from modules.utils import scale_gradient

        # Parameters from first module
        config = self.loss_modules[0].config
        device = self.loss_modules[0].device

        # Convert NamedTuples/dataclasses to dicts if necessary
        if isinstance(predictions, dict):
            pass
        elif hasattr(predictions, "_asdict"):
            predictions = predictions._asdict()
        else:
            predictions = vars(predictions)
        targets = targets if isinstance(targets, dict) else vars(targets)
        # Determine actual batch size B from tensors
        B = 0
        for val in predictions.values():
            if torch.is_tensor(val):
                B = val.shape[0]
                break
        if B == 0:
            for val in targets.values():
                if torch.is_tensor(val):
                    B = val.shape[0]
                    break

        if B == 0:
            # Fallback (should not happen if inputs are valid)
            B = config.minibatch_size

        if weights is None:
            weights = torch.ones(B, device=device)

        if gradient_scales is None:
            gradient_scales = torch.ones((1, 1), device=device)

        total_loss_dict = {
            module.optimizer_name: torch.tensor(0.0, device=device)
            for module in self.loss_modules
        }
        loss_dict = {module.name: 0.0 for module in self.loss_modules}
        elementwise_losses = {}

        # --- Vectorized Loss Pass ---
        for module in self.loss_modules:
            # Compute unreduced elementwise loss: [B, T, ...]
            loss_bt = module.compute_loss(predictions, targets, context)

            # Apply mask [B, T]
            mask_bt = module.get_mask(context)
            loss_bt = loss_bt * mask_bt

            # Apply gradient scaling and PER weights via broadcasting
            # loss_bt: [B, T]
            # gradient_scales: [1, T] -> broadcast to [B, T]
            # weights: [B] -> reshape to [B, 1] -> broadcast to [B, T]
            
            # Align shapes for broadcasting
            scales = gradient_scales # [1, T]
            w = weights.view(-1, 1) # [B, 1]
            
            weighted_scaled_loss = loss_bt * scales * w

            # Accumulate for optimizer
            total_loss_dict[module.optimizer_name] += weighted_scaled_loss.sum()

            # Accumulate for logging (unweighted)
            loss_dict[module.name] += loss_bt.detach().sum().item()

        # Average the total loss and logging losses by batch size
        minibatch_size = config.minibatch_size
        for opt_name in total_loss_dict:
            total_loss_dict[opt_name] = total_loss_dict[opt_name] / minibatch_size

        # Average the total loss by batch size
        for opt_name in total_loss_dict:
            total_loss_dict[opt_name] = (
                total_loss_dict[opt_name] / config.minibatch_size
            )

        # Average accumulated losses for logging
        for key in loss_dict:
            loss_dict[key] /= config.minibatch_size

        # --- 4. Propagate auxiliary metrics from context (e.g., approx_kl) ---
        for key, value in context.items():
            if key == "full_targets" or key == "target_values_next":
                continue
            if (
                isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], (int, float))
            ):
                loss_dict[key] = float(np.mean(value))
            elif isinstance(value, (int, float)):
                loss_dict[key] = float(value)

        # 4. Final Priority Computation (Standalone Strategy)
        priorities = None
        if self.priority_computer is not None:
            priorities = self.priority_computer(
                predictions, targets, elementwise_losses
            )

        return total_loss_dict, loss_dict, priorities
