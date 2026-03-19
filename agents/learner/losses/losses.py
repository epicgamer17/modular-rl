import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
from torch import Tensor
from modules.world_models.inference_output import InferenceOutput
from utils.telemetry import append_metric


class LossModule(ABC):
    """
    Unified base class for all loss modules.
    Works for both single-step (DQN, C51) and sequence (MuZero) losses.
    """

    def __init__(
        self,
        config,
        device,
        optimizer_name: str = "default",
        mask_key: str = "value_mask",
    ):
        self.config = config
        self.device = device
        self.optimizer_name = optimizer_name
        self.mask_key = mask_key
        self.name = self.__class__.__name__

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

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        """Determine if this loss should be computed for the batch."""
        return True

    def get_mask(self, targets: dict) -> torch.Tensor:
        """Get the mask to apply for this loss (B, T)."""
        mask = targets.get(self.mask_key)
        if mask is None:
            raise KeyError(
                f"Missing required mask '{self.mask_key}' for {self.__class__.__name__}"
            )
        return mask

    @abstractmethod
    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """
        Compute elementwise loss for the entire sequence.

        Returns:
            elementwise_tensor of shape (B, T)
        """
        pass  # pragma: no cover

    def compute_priority(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> Optional[torch.Tensor]:
        """
        Calculates PER priorities for the sequence.
        Returns:
            Priority tensor of shape (B,) or None.
        """
        return None  # pragma: no cover


# ============================================================================
# OLD DQN-STYLE LOSSES (Updated to work with unified interface)
# ============================================================================


class StandardDQNLoss(LossModule):
    def __init__(
        self,
        config,
        device,
        action_selector: Optional[object] = None,
        optimizer_name: str = "default",
        mask_key: str = "value_mask",
    ):
        super().__init__(
            config, device, optimizer_name=optimizer_name, mask_key=mask_key
        )
        self.action_selector = action_selector

    @property
    def required_predictions(self) -> set[str]:
        return {"q_values"}

    @property
    def required_targets(self) -> set[str]:
        return {"q_values", "actions"}

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict
    ) -> torch.Tensor:
        q_values = predictions["q_values"]
        actions = targets["actions"].long()

        # 1. Capture and Validate Shapes
        assert q_values.ndim == 3, f"StandardDQNLoss requires [B, T, Actions] predictions, got {q_values.shape}"
        assert actions.ndim == 2, f"StandardDQNLoss requires [B, T] action targets, got {actions.shape}"
        B, T, num_actions = q_values.shape

        # 2. Flatten for vectorized indexing
        flat_q = q_values.reshape(-1, num_actions)
        flat_actions = actions.reshape(-1)

        # [B * T]
        selected_q = flat_q[torch.arange(B * T, device=self.device), flat_actions]
        targets_val = targets["q_values"].reshape(-1)

        # 3. Compute and Unflatten
        assert selected_q.shape == targets_val.shape, f"StandardDQNLoss: shape mismatch between selected_q {selected_q.shape} and targets_val {targets_val.shape}"
        loss = self.config.loss_function(selected_q, targets_val, reduction="none")
        return loss.reshape(B, T)

    def compute_priority(self, predictions: dict, targets: dict, context: dict) -> torch.Tensor:
        q_values = predictions["q_values"]
        actions = targets["actions"].long()
        target_q = targets["q_values"]
        B, T = actions.shape[:2]

        # Q-value of the chosen action [B, T]
        flat_q = q_values.reshape(-1, q_values.shape[-1])
        pred_q = flat_q[torch.arange(B * T, device=self.device), actions.reshape(-1)].reshape(B, T)

        # Return max TD-error over the sequence for each batch element [B]
        assert target_q.shape == pred_q.shape, f"StandardDQNLoss priority: shape mismatch between target_q {target_q.shape} and pred_q {pred_q.shape}"
        td_error = torch.abs(target_q - pred_q).detach()
        return td_error.max(dim=1).values


class C51Loss(LossModule):
    def __init__(
        self,
        config,
        device,
        action_selector: Optional[object] = None,
        optimizer_name: str = "default",
        mask_key: str = "value_mask",
    ):
        super().__init__(
            config, device, optimizer_name=optimizer_name, mask_key=mask_key
        )
        self.action_selector = action_selector
        self.support = torch.linspace(
            self.config.v_min,
            self.config.v_max,
            self.config.atom_size,
            device=self.device,
        )

    @property
    def required_predictions(self) -> set[str]:
        return {"q_logits"}

    @property
    def required_targets(self) -> set[str]:
        return {"q_logits", "actions"}

    def _project_target_distribution(
        self,
        rewards: torch.Tensor,
        terminal_mask: torch.Tensor,
        next_probs: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = rewards.shape[0]
        discount = self.config.discount_factor**self.config.n_step
        delta_z = (self.config.v_max - self.config.v_min) / (self.config.atom_size - 1)

        # Compute the projected support: Tz = r + gamma * z
        tz = (
            rewards.view(-1, 1)
            + discount * (~terminal_mask.bool()).view(-1, 1) * self.support.view(1, -1)
        ).clamp(self.config.v_min, self.config.v_max)

        # Map back to index space: b = (Tz - v_min) / delta_z
        b = (tz - self.config.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        dist_l = u.float() - b
        dist_u = b - l.float()

        mask_equal = l == u
        dist_l[mask_equal] = 1.0
        dist_u[mask_equal] = 0.0

        projected = torch.zeros((batch_size, self.config.atom_size), device=self.device)
        projected.scatter_add_(1, l, next_probs * dist_l)
        projected.scatter_add_(1, u, next_probs * dist_u)

        return projected

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict
    ) -> torch.Tensor:
        online_q_logits = predictions["q_logits"]
        actions = targets["actions"].long()
        target_q_logits = targets["q_logits"]

        # 1. Capture and Validate
        assert online_q_logits.ndim == 4, f"C51Loss requires [B, T, Actions, Atoms], got {online_q_logits.shape}"
        B, T, num_actions, atoms = online_q_logits.shape

        # 2. Flatten and select
        flat_logits = online_q_logits.reshape(-1, num_actions, atoms)
        flat_actions = actions.reshape(-1)
        
        # [B * T, Atoms]
        chosen_logits = flat_logits[torch.arange(B * T, device=self.device), flat_actions]
        log_probs = F.log_softmax(chosen_logits, dim=-1)
        
        # [B * T, Atoms]
        flat_target_q_logits = target_q_logits.reshape(-1, atoms)
        
        # [B * T]
        assert flat_target_q_logits.shape == log_probs.shape, f"C51Loss: shape mismatch between flat_target_q_logits {flat_target_q_logits.shape} and log_probs {log_probs.shape}"
        loss = -(flat_target_q_logits * log_probs).sum(dim=-1)
        return loss.reshape(B, T)

    def compute_priority(self, predictions: dict, targets: dict, context: dict) -> torch.Tensor:
        # Predict Expected Q
        probs = torch.softmax(predictions["q_logits"], dim=-1)
        support = self.support.to(device=probs.device, dtype=probs.dtype)
        q_values = (probs * support).sum(dim=-1) # [B, T, Actions]
        
        actions = targets["actions"].long()
        B, T = actions.shape[:2]
        
        flat_q = q_values.reshape(-1, q_values.shape[-1])
        pred_q = flat_q[torch.arange(B * T, device=self.device), actions.reshape(-1)].reshape(B, T)

        # Target Expected Q [B, T]
        target_q = (targets["q_logits"] * support).sum(dim=-1)
        assert target_q.shape == pred_q.shape, f"C51Loss priority: shape mismatch between target_q {target_q.shape} and pred_q {pred_q.shape}"

        return torch.abs(target_q - pred_q).detach().max(dim=1).values


# ============================================================================
# NEW MUZERO-STYLE LOSSES (Updated to work with unified interface)
# ============================================================================


class ValueLoss(LossModule):
    """Value prediction loss module."""

    def __init__(
        self,
        config,
        device,
        optimizer_name: str = "default",
        mask_key: str = "value_mask",
    ):
        super().__init__(
            config, device, optimizer_name=optimizer_name, mask_key=mask_key
        )

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

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        return True

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        values = predictions["values"]
        target_values = targets["values"]
        
        # 1. Capture and Validate
        assert values.ndim >= 2, f"ValueLoss requires [B, T, ...] predictions, got {values.shape}"
        assert target_values.ndim >= 2, f"ValueLoss requires [B, T, ...] targets, got {target_values.shape}"
        B, T = target_values.shape[:2]

        # 2. Conversion and Support Logic
        if self.config.support_range is not None:
            from modules.utils import scalar_to_support
            # [B, T] -> [B, T, atoms]
            target_values = scalar_to_support(target_values, self.config.support_range).to(self.device)
            predicted_values = values
        else:
            # Ensure both are [B, T]
            predicted_values = values.reshape(B, T)
            target_values = target_values.reshape(B, T)

        # 3. Compute Vectorized Loss [B, T]
        assert predicted_values.shape == target_values.shape, f"ValueLoss: shape mismatch between predicted_values {predicted_values.shape} and target_values {target_values.shape}"
        value_loss = self.config.value_loss_function(predicted_values, target_values, reduction="none")
        if value_loss.ndim > 2:
            value_loss = value_loss.sum(dim=-1)

        # 4. Latent visualization (root only metadata)
        if getattr(self.config, "latent_viz_interval", 0) > 0:
            viz_interval = self.config.latent_viz_interval
            if context.get("training_step", 0) % viz_interval == 0:
                 from utils.telemetry import add_latent_visualization_metric
                 # UsePredictions: 'latents' root is expected at index 0
                 s0 = predictions["latents"][:, 0]
                 actions = targets.get("actions")
                 if actions is not None:
                      metrics = context.setdefault("metrics", {})
                      add_latent_visualization_metric(
                          metrics, "latent_root", s0.detach().cpu(),
                          labels=actions[:, 0].detach().cpu(),
                          method=getattr(self.config, "latent_viz_method", "pca")
                      )

        return self.config.value_loss_factor * value_loss

    def compute_priority(self, predictions: dict, targets: dict, context: dict) -> torch.Tensor:
        from modules.utils import support_to_scalar

        values = predictions["values"]
        target_values = targets["values"]
        B, T = target_values.shape[:2]

        if self.config.support_range is not None:
            # Flatten to use support_to_scalar [N, atoms] -> [N]
            flat_values = values.reshape(-1, values.shape[-1])
            flat_pred = support_to_scalar(flat_values, self.config.support_range)
            pred_scalar = flat_pred.reshape(B, T)
            target_scalar = target_values.reshape(B, T)
        else:
            pred_scalar = values.reshape(B, T)
            target_scalar = target_values.reshape(B, T)

        # Return max TD-error over the sequence [B]
        assert target_scalar.shape == pred_scalar.shape, f"ValueLoss priority: shape mismatch between target_scalar {target_scalar.shape} and pred_scalar {pred_scalar.shape}"
        return torch.abs(target_scalar - pred_scalar).detach().max(dim=1).values


class PolicyLoss(LossModule):
    """Policy prediction loss module."""

    def __init__(
        self,
        config,
        device,
        optimizer_name: str = "default",
        mask_key: str = "policy_mask",
    ):
        super().__init__(
            config, device, optimizer_name=optimizer_name, mask_key=mask_key
        )

    @property
    def required_predictions(self) -> set[str]:
        return {"policies"}

    @property
    def required_targets(self) -> set[str]:
        return {"policies"}

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        return True

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        policies = predictions["policies"]
        target_policies = targets["policies"]
        
        # 1. Capture and Validate
        assert policies.shape == target_policies.shape, f"PolicyLoss shape mismatch: preds {policies.shape} vs targets {target_policies.shape}"
        B, T, num_actions = policies.shape

        # 2. Vectorized Loss calculation [B, T]
        if self.config.policy_loss_function == F.kl_div:
            log_probs = F.log_softmax(policies, dim=-1)
            assert log_probs.shape == target_policies.shape, f"PolicyLoss KL: shape mismatch between log_probs {log_probs.shape} and target_policies {target_policies.shape}"
            policy_loss = self.config.policy_loss_function(
                log_probs, target_policies, reduction="none"
            )
            if policy_loss.ndim > 2:
                policy_loss = policy_loss.sum(dim=-1)
        else:
            # Flatten to [N, num_actions] because some cross_entropy versions prefer 2D/3D but not 4D?
            # actually cross_entropy should handle it, but flattening is safest.
            p_flat = policies.reshape(-1, num_actions)
            tp_flat = target_policies.reshape(-1, num_actions)
            assert p_flat.shape == tp_flat.shape, f"PolicyLoss: shape mismatch between flattened policies {p_flat.shape} and target_policies {tp_flat.shape}"
            policy_loss = self.config.policy_loss_function(
                p_flat, 
                tp_flat, 
                reduction="none"
            )
            policy_loss = policy_loss.reshape(B, T)

        return policy_loss


class RewardLoss(LossModule):
    """Reward prediction loss module."""

    def __init__(
        self,
        config,
        device,
        optimizer_name: str = "default",
        mask_key: str = "reward_mask",
    ):
        super().__init__(
            config, device, optimizer_name=optimizer_name, mask_key=mask_key
        )

    @property
    def required_predictions(self) -> set[str]:
        return {"rewards"}

    @property
    def required_targets(self) -> set[str]:
        return {"rewards"}

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        return True

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        rewards = predictions["rewards"]
        target_rewards = targets["rewards"]

        # 1. Capture and Validate
        assert rewards.ndim >= 2, f"RewardLoss requires [B, T, ...] predictions, got {rewards.shape}"
        assert target_rewards.ndim >= 2, f"RewardLoss requires [B, T, ...] targets, got {target_rewards.shape}"
        B, T = target_rewards.shape[:2]

        # 2. Conversion and Support Logic
        if self.config.support_range is not None:
            from modules.utils import scalar_to_support
            # Predictions: [B, T, atoms], Targets: [B, T, atoms]
            target_dist = scalar_to_support(target_rewards, self.config.support_range).to(self.device)
            assert rewards.shape == target_dist.shape, f"RewardLoss Support: shape mismatch between rewards {rewards.shape} and target_dist {target_dist.shape}"
            reward_loss = self.config.reward_loss_function(rewards, target_dist, reduction="none")
            if reward_loss.ndim > 2:
                reward_loss = reward_loss.sum(dim=-1)
        else:
            # Standard Scalar MSE: Both [B, T]
            preds_r = rewards.reshape(B, T)
            target_r = target_rewards.reshape(B, T)
            assert preds_r.shape == target_r.shape, f"RewardLoss Scalar: shape mismatch between rewards {preds_r.shape} and target_rewards {target_r.shape}"
            reward_loss = self.config.reward_loss_function(
                preds_r, target_r, reduction="none"
            )

        return reward_loss


class ToPlayLoss(LossModule):
    """To-play (turn indicator) prediction loss module."""

    def __init__(
        self,
        config,
        device,
        optimizer_name: str = "default",
        mask_key: str = "to_play_mask",
    ):
        super().__init__(
            config, device, optimizer_name=optimizer_name, mask_key=mask_key
        )

    @property
    def required_predictions(self) -> set[str]:
        return {"to_plays"}

    @property
    def required_targets(self) -> set[str]:
        return {"to_plays"}

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        # Only compute for multi-player games
        return self.config.game.num_players != 1

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        to_plays = predictions["to_plays"]
        target_to_plays = targets["to_plays"]

        # 1. Capture and Validate
        assert to_plays.ndim == 3, f"ToPlayLoss requires [B, T, num_players], got {to_plays.shape}"
        assert target_to_plays.ndim == 2, f"ToPlayLoss requires [B, T] targets, got {target_to_plays.shape}"
        B, T = target_to_plays.shape

        # 2. Compute Vectorized Cross Entropy [B, T]
        # Flatten to use standard functional interface
        num_players = self.config.game.num_players
        p_flat = to_plays.reshape(-1, num_players)
        tp_flat = target_to_plays.reshape(-1)
        # For cross entropy, labels [N] match logits [N, C] prefix
        assert p_flat.shape[0] == tp_flat.shape[0], f"ToPlayLoss: count mismatch between predictions {p_flat.shape[0]} and targets {tp_flat.shape[0]}"
        to_play_loss = self.config.to_play_loss_function(
            p_flat,
            tp_flat.long(),
            reduction="none"
        ).reshape(B, T)

        return self.config.to_play_loss_factor * to_play_loss


class RelativeToPlayLoss(LossModule):
    """
    To-play loss for relative turn shifts (ΔP).
    Calculates ΔP targets from the sequence of absolute player indices:
    ΔP_k = (P_k - P_{k-1}) mod num_players.
    """

    def __init__(
        self,
        config,
        device,
        optimizer_name: str = "default",
        mask_key: str = "to_play_mask",
    ):
        super().__init__(
            config, device, optimizer_name=optimizer_name, mask_key=mask_key
        )

    @property
    def required_predictions(self) -> set[str]:
        return {"to_plays"}

    @property
    def required_targets(self) -> set[str]:
        # Needs to_plays to calculate delta
        return {"to_plays"}

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        return self.config.game.num_players > 1

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        # predictions["to_plays"] contains ΔP logits for seq (B, T, num_players)
        # Note: In MuZero, ΔP is usually predicted for k=1..T-1.
        delta_p_logits = predictions["to_plays"]
        full_to_plays = targets["to_plays"]
        
        # 1. Capture and Validate
        assert delta_p_logits.ndim == 3, f"RelativeToPlayLoss requires [B, T, num_players], got {delta_p_logits.shape}"
        assert full_to_plays.ndim == 2, f"RelativeToPlayLoss requires [B, T] to_plays, got {full_to_plays.shape}"
        B, T = full_to_plays.shape
        num_players = self.config.game.num_players

        # 2. Calculate ΔP targets: [B, T-1]
        p_prev = full_to_plays[:, :-1]
        p_curr = full_to_plays[:, 1:]
        target_delta_p = (p_curr - p_prev) % num_players
        
        # 3. Align and Compute Vectorized Cross Entropy
        # ΔP for k=1 corresponds to predictions index k=1 (if root is k=0)
        # Actually MuZero predicts p_k from s_k. s_k is produced by step(s_{k-1}, a_{k-1}).
        # So delta_p_logits[:, 1] is the prediction for ΔP_1 = (P1 - P0).
        
        # Zero-pad or slice to match T?
        # Standard approach: only compute for indices where we have a prev player.
        # [B, T-1]
        preds_p = delta_p_logits[:, 1:].reshape(-1, num_players)
        targets_p = target_delta_p.reshape(-1)
        assert preds_p.shape[0] == targets_p.shape[0], f"RelativeToPlayLoss: count mismatch between predictions {preds_p.shape[0]} and targets {targets_p.shape[0]}"
        loss_flat = self.config.to_play_loss_function(
            preds_p,
            targets_p.long(),
            reduction="none"
        )
        
        # Reshape and pad first index (k=0) with 0.0 to maintain [B, T] shape
        loss_seq = loss_flat.reshape(B, T-1)
        zero_pad = torch.zeros((B, 1), device=self.device)
        total_loss = torch.cat([zero_pad, loss_seq], dim=1)
        
        return self.config.to_play_loss_factor * total_loss

        # Loss calculation
        loss = self.config.to_play_loss_factor * self.config.to_play_loss_function(
            delta_p_logits_k, target_delta_p_k, reduction="none"
        )

        return loss


class ConsistencyLoss(LossModule):
    """Consistency loss module (EfficientZero style)."""

    def __init__(self, config, device, agent_network, optimizer_name: str = "default", mask_key: str = "policy_mask"):
        super().__init__(config, device, optimizer_name=optimizer_name, mask_key=mask_key)
        self.agent_network = agent_network

    @property
    def required_predictions(self) -> set[str]:
        return {"latents"}

    @property
    def required_targets(self) -> set[str]:
        return {"consistency_targets"}

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        return True

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        latent_states = predictions["latents"]
        target_features = targets["consistency_targets"]
        
        # 1. Capture and Validate
        assert latent_states.ndim >= 3, f"ConsistencyLoss requires [B, T, D], got {latent_states.shape}"
        assert target_features.ndim >= 2, f"ConsistencyLoss requires [B, T, D], got {target_features.shape}"
        B, T = target_features.shape[:2]

        if isinstance(latent_states, dict):
            latent_states = latent_states["dynamics"]

        # 2. Process Vectorized projection on flattened batch-time
        flat_latents = latent_states.reshape(-1, latent_states.shape[-1])
        proj_preds = self.agent_network.project(flat_latents, grad=True)
        f2 = F.normalize(proj_preds, p=2.0, dim=-1, eps=1e-5)

        # 3. Compute Vectorized Similarity Loss [B, T]
        flat_targets = target_features.reshape(-1, target_features.shape[-1])
        assert flat_targets.shape == f2.shape, f"ConsistencyLoss: shape mismatch between targets {flat_targets.shape} and predictions {f2.shape}"
        flat_consistency = -(flat_targets * f2).sum(dim=1)
        
        return self.config.consistency_loss_factor * flat_consistency.reshape(B, T)


class ChanceQLoss(LossModule):
    """Q-value loss for chance nodes (stochastic MuZero)."""

    def __init__(self, config, device, optimizer_name: str = "default"):
        super().__init__(config, device, optimizer_name=optimizer_name)

    @property
    def required_predictions(self) -> set[str]:
        return {"chance_values"}

    @property
    def required_targets(self) -> set[str]:
        # Uses target_values_next which is targets["values"][:, k]
        return {"values"}

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        return self.config.stochastic

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        chance_values = predictions["chance_values"]
        # In vectorized mode, LossPipeline should provide target_values_next as [B, T]
        target_chance_values = context.get("target_values_next")
        
        # 1. Capture and Validate
        assert chance_values.ndim >= 2, f"ChanceQLoss requires [B, T, ...] predictions, got {chance_values.shape}"
        assert target_chance_values is not None, "ChanceQLoss requires 'target_values_next' in context"
        B, T = target_chance_values.shape[:2]

        # 2. Conversion and Support Logic
        if self.config.support_range is not None:
            from modules.utils import scalar_to_support
            # [B, T] -> [B, T, atoms]
            target_dist = scalar_to_support(target_chance_values, self.config.support_range).to(self.device)
            predicted_chance_values = chance_values
        else:
            predicted_chance_values = chance_values.reshape(B, T)
            target_dist = target_chance_values.reshape(B, T)

        # 3. Compute Vectorized Loss [B, T]
        assert predicted_chance_values.shape == target_dist.shape, f"ChanceQLoss: shape mismatch between predictions {predicted_chance_values.shape} and targets {target_dist.shape}"
        q_loss = self.config.value_loss_function(predicted_chance_values, target_dist, reduction="none")
        if q_loss.ndim > 2:
            q_loss = q_loss.sum(dim=-1)

        # 4. Entropy metrics for full sequence
        if self.config.support_range is not None:
            metrics = context.setdefault("metrics", {})
            probs = torch.softmax(predicted_chance_values, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1) # [B, T]
            mask = self.get_mask(targets).bool()
            valid_entropy = entropy[mask]
            append_metric(metrics, "chance_q_entropy", valid_entropy.mean().item() if valid_entropy.numel() > 0 else 0.0)

        return self.config.value_loss_factor * q_loss


class SigmaLoss(LossModule):
    """Sigma (chance code prediction) loss for stochastic MuZero."""

    def __init__(self, config, device, optimizer_name: str = "default"):
        super().__init__(config, device, optimizer_name=optimizer_name)

    @property
    def required_predictions(self) -> set[str]:
        return {"chance_logits"}

    @property
    def required_targets(self) -> set[str]:
        return {"chance_codes"}

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        return self.config.stochastic

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        chance_logits = predictions["chance_logits"]
        target_codes = targets["chance_codes"].reshape(-1).long() # Flatten for indexing

        # 1. Capture and Validate
        assert chance_logits.ndim == 3, f"SigmaLoss requires [B, T, codebook_size], got {chance_logits.shape}"
        B, T, codebook_size = chance_logits.shape

        # 2. Compute Vectorized Loss [B, T]
        if self.config.sigma_loss == F.cross_entropy:
            p_flat = chance_logits.reshape(-1, codebook_size)
            assert p_flat.shape[0] == target_codes.shape[0], f"SigmaLoss CrossEntropy: count mismatch between predictions {p_flat.shape[0]} and targets {target_codes.shape[0]}"
            sigma_loss_flat = self.config.sigma_loss(
                p_flat, target_codes, reduction="none"
            )
        else:
            target_onehot = F.one_hot(target_codes, num_classes=codebook_size).float()
            p_flat = chance_logits.reshape(-1, codebook_size)
            assert p_flat.shape == target_onehot.shape, f"SigmaLoss MSE/Other: shape mismatch between predictions {p_flat.shape} and target_onehot {target_onehot.shape}"
            sigma_loss_flat = self.config.sigma_loss(
                p_flat, target_onehot.detach(), reduction="none"
            )

        # 3. Metrics for full sequence
        metrics = context.setdefault("metrics", {})
        probs = torch.softmax(chance_logits, dim=-1)
        mask = self.get_mask(targets).bool() # [B, T]
        valid_probs = probs[mask]
        if valid_probs.shape[0] > 0:
            append_metric(metrics, "chance_probs", valid_probs.mean(dim=0))
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1) # [B, T]
        valid_entropy = entropy[mask]
        append_metric(metrics, "chance_entropy", valid_entropy.mean().item() if valid_entropy.numel() > 0 else 0.0)
        
        valid_codes = targets["chance_codes"][mask]
        append_metric(metrics, "num_codes", int(torch.unique(valid_codes).numel()))

        return sigma_loss_flat.reshape(B, T)


class VQVAECommitmentLoss(LossModule):
    """VQ-VAE commitment cost for encoder (stochastic MuZero)."""

    def __init__(self, config, device, optimizer_name: str = "default"):
        super().__init__(config, device, optimizer_name=optimizer_name)

    @property
    def required_predictions(self) -> set[str]:
        return {"chance_encoder_embeddings"}

    @property
    def required_targets(self) -> set[str]:
        return {"chance_codes"}

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        return self.config.stochastic and not self.config.use_true_chance_codes

    def get_mask(self, targets: dict) -> torch.Tensor:
        return super().get_mask(targets)

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        embeddings = predictions["chance_encoder_embeddings"]
        target_codes = targets["chance_codes"].squeeze(-1).long()
        
        # 1. Capture and Validate
        assert embeddings.ndim == 3, f"VQVAECommitmentLoss requires [B, T, D], got {embeddings.shape}"
        B, T, D = embeddings.shape
        
        # 2. Vectorized one-hot and diff
        # [B, T, D]
        target_onehot = F.one_hot(target_codes, num_classes=D).float()
        
        # [B, T, D] -> [B, T]
        assert embeddings.shape == target_onehot.shape, f"VQVAECommitmentLoss: shape mismatch between embeddings {embeddings.shape} and target_onehot {target_onehot.shape}"
        diff = embeddings - target_onehot.detach()
        commitment_cost = torch.sum(diff.pow(2), dim=-1)
        
        return self.config.vqvae_commitment_cost_factor * commitment_cost


class PPOPolicyLoss(LossModule):
    def __init__(
        self,
        config,
        device,
        clip_param: float,
        entropy_coefficient: float,
        policy_strategy: Optional[object] = None,
        optimizer_name: str = "default",
    ):
        super().__init__(config, device, optimizer_name=optimizer_name)
        self.clip_param = clip_param
        self.entropy_coefficient = entropy_coefficient
        self.policy_strategy = policy_strategy

    @property
    def required_predictions(self) -> set[str]:
        return {"policies"}

    @property
    def required_targets(self) -> set[str]:
        return {"actions", "old_log_probs", "advantages"}

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        return True

    def compute_loss(self, predictions: dict, targets: dict, context: dict) -> torch.Tensor:
        """PPO Policy Loss: returns [B, T]"""
        policy_logits = predictions["policies"]
        actions = targets["actions"]
        old_log_probs = targets["old_log_probs"]
        advantages = targets["advantages"]
        
        # 1. Capture and Validate
        assert policy_logits.ndim == 3, f"PPOPolicyLoss requires [B, T, A], got {policy_logits.shape}"
        B, T = old_log_probs.shape[:2]

        # 2. Vectorized Distribution math
        if self.policy_strategy is not None:
             # Strategy should handle [B, T, A] -> Dist with [B, T] sample shapes
             dist = self.policy_strategy.get_distribution(policy_logits)
        else:
             dist = torch.distributions.Categorical(logits=policy_logits)
        
        # [B, T]
        log_probs = dist.log_prob(actions)
        assert log_probs.shape == old_log_probs.shape, f"PPOPolicyLoss: shape mismatch between log_probs {log_probs.shape} and old_log_probs {old_log_probs.shape}"
        ratio = torch.exp(log_probs - old_log_probs)
        
        assert ratio.shape == advantages.shape, f"PPOPolicyLoss ratio vs advantages: {ratio.shape} vs {advantages.shape}"
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        
        entropy = dist.entropy()
        
        loss = -torch.min(surr1, surr2) - self.entropy_coefficient * entropy
        
        # 3. Stats for full sequence
        with torch.no_grad():
             approx_kl = (old_log_probs - log_probs).mean()
             if "approx_kl" not in context:
                  context["approx_kl"] = []
             context["approx_kl"].append(approx_kl.item())
             
        return loss
        # Note: mask is applied in LossPipeline.run
        # but for PPO it's usually 1s.


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
        optimizer_name: str = "default",
    ):
        super().__init__(config, device, optimizer_name=optimizer_name)
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

        if value_logits.ndim == 1:
            return value_logits

        if value_logits.shape[-1] == 1:
            return value_logits.squeeze(-1)

        if self.atom_size > 1 and self.v_min is not None and self.v_max is not None:
            support = torch.linspace(
                self.v_min,
                self.v_max,
                value_logits.shape[-1],
                device=value_logits.device,
                dtype=value_logits.dtype,
            )
            probs = torch.softmax(value_logits, dim=-1)
            return (probs * support).sum(dim=-1)

        raise ValueError(
            "PPOValueLoss received multi-logit values without distributional bounds "
            "(v_min/v_max)."
        )

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict
    ) -> torch.Tensor:
        """PPO Value Loss: returns [B, T]"""
        value_logits = predictions["values"]
        returns = targets["returns"]
        
        # 1. Capture and Validate
        assert value_logits.ndim >= 2, f"PPOValueLoss requires [B, T, ...], got {value_logits.shape}"
        B, T = returns.shape[:2]
        
        # 2. Vectorized Conversion
        # Flatten to use existing _to_scalar_values or wrap it
        flat_logits = value_logits.reshape(-1, value_logits.shape[-1])
        flat_values = self._to_scalar_values(flat_logits)
        values = flat_values.reshape(B, T)
        
        # 3. Compute Vectorized MSE [B, T]
        assert returns.shape == values.shape, f"PPOValueLoss: shape mismatch between returns {returns.shape} and values {values.shape}"
        return self.critic_coefficient * ((returns - values) ** 2)


class ImitationLoss(LossModule):
    def __init__(
        self, config, device, num_actions: int, optimizer_name: str = "default"
    ):
        super().__init__(config, device, optimizer_name=optimizer_name)
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
        """Imitation Loss: returns [B, T]"""
        policy_logits = predictions["policies"]
        target_policies = targets["target_policies"]
        
        # 1. Capture and Validate
        assert policy_logits.ndim == 3, f"ImitationLoss requires [B, T, A], got {policy_logits.shape}"
        B, T, A = policy_logits.shape
        
        # 2. Vectorized conversion to one-hot if labels provided
        if target_policies.ndim < 3:
            # Assume [B, T] of labels
            one_hot = F.one_hot(target_policies.long(), num_classes=A).float()
            target_policies = one_hot

        # 3. Metrics for full sequence
        probs = torch.softmax(policy_logits, dim=-1)
        metrics = context.setdefault("metrics", {})
        append_metric(metrics, "sl_policy", probs.mean(dim=(0, 1)).detach().cpu())

        # 4. Compute Vectorized Loss [B, T]
        p_flat = policy_logits.reshape(-1, A)
        tp_flat = target_policies.reshape(-1, A)
        assert p_flat.shape == tp_flat.shape, f"ImitationLoss: shape mismatch between predictions {p_flat.shape} and target_policies {tp_flat.shape}"
        loss = self.loss_function(
            p_flat, 
            tp_flat
        ).reshape(B, T)
        
        if loss.ndim > 2:
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

    def __init__(self, modules: list[LossModule]):
        self.modules = modules

    def validate_dependencies(
        self, network_output_keys: set[str], target_keys: set[str]
    ) -> None:
        """
        Verify that the provided keys satisfy all module requirements.
        Raises ValueError with detailed error message on failure.
        """
        for module in self.modules:
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
        Run the loss pipeline in a single vectorized pass across all sequence steps.
        """
        from modules.utils import support_to_scalar, scale_gradient
        from agents.learner.losses.shape_validator import ShapeValidator
        import numpy as np

        # 1. Config and Shape Validation
        config = self.modules[0].config
        device = self.modules[0].device
        ShapeValidator(config).validate(predictions, targets)

        # 2. Key/Format Normalization
        if not isinstance(predictions, dict):
            predictions = getattr(predictions, "_asdict", lambda: vars(predictions))()
        if not isinstance(targets, dict):
            targets = targets if isinstance(targets, dict) else vars(targets)

        # Determine dimensions B and T
        first_tensor = next(iter(predictions.values()))
        B, T = first_tensor.shape[:2]

        # 3. Defaults and Scaling
        if weights is None:
            weights = torch.ones(B, device=device)
        if gradient_scales is None:
            gradient_scales = torch.ones((1, T), device=device)

        context["full_targets"] = targets
        # Vectorized ChanceQ targets: shift values by 1
        if "values" in targets:
            v = targets["values"]
            v_next = torch.zeros_like(v)
            v_next[:, :-1] = v[:, 1:]
            context["target_values_next"] = v_next

        total_loss_dict = {
            m.optimizer_name: torch.tensor(0.0, device=device) for m in self.modules
        }
        loss_dict = {}
        priorities = torch.zeros(B, device=device)

        # 4. Single-Pass Vectorized Execution
        for module in self.modules:
            if not module.should_compute(predictions, targets, context):
                continue

            # Compute [B, T] elementwise loss
            elementwise_loss = module.compute_loss(predictions, targets, context)
            mask = module.get_mask(targets)
            
            # Apply Gradient Scaling [1, T] -> [B, T]
            scaled_loss = scale_gradient(elementwise_loss, gradient_scales)
            
            # Mask and Reduce over sequence [B, T] -> [B]
            loss_seq_weighted = (scaled_loss * mask.float()).sum(dim=1)
            
            # Apply PER Weights [B]
            weighted_batch_loss = loss_seq_weighted * weights
            
            # Final Mean for this batch
            total_scalar_loss = weighted_batch_loss.mean()
            
            total_loss_dict[module.optimizer_name] += total_scalar_loss
            loss_dict[module.name] = total_scalar_loss.item()

            # PER Priorities [B]
            if hasattr(module, "compute_priority"):
                p = module.compute_priority(predictions, targets, context)
                if p is not None:
                    priorities = torch.max(priorities, p)

        # 5. Extract Auxiliary metrics from context (approx_kl, etc.)
        for key, value in context.items():
            if key in ["full_targets", "target_values_next", "has_valid_action_mask", "is_same_game"]:
                continue
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                loss_dict[key] = float(np.mean(value))

        return total_loss_dict, loss_dict, priorities
