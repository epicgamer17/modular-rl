import torch
import torch.nn.functional as F
from typing import Any
from agents.learner.losses.base import BaseLoss

class QBootstrappingLoss(BaseLoss):
    """
    Standard TD target loss for Q-learning.
    Indexes the prediction tensor by the taken actions to compute TD errors.
    """
    def __init__(
        self,
        config,
        device,
        representation: Any,
        optimizer_name: str = "default",
        mask_key: str = "value_mask",
    ):
        # 1. Determine Pred/Target keys and default Loss function based on atom_size
        is_categorical = getattr(config, "atom_size", 1) > 1
        pred_key = "q_logits" if is_categorical else "q_values"
        target_key = "q_logits" if is_categorical else "q_values"

        # Use cross_entropy for C51, but allow override via config.loss_function (MSE)
        default_loss_fn = F.cross_entropy if is_categorical else F.mse_loss
        loss_fn = getattr(config, "loss_function", default_loss_fn)

        super().__init__(
            config=config,
            device=device,
            pred_key=pred_key,
            target_key=target_key,
            mask_key=mask_key,
            representation=representation,
            loss_fn=loss_fn,
            optimizer_name=optimizer_name,
        )

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict
    ) -> torch.Tensor:
        q_preds = predictions[self.pred_key]
        actions = targets["actions"].long()

        # 1. Capture and Validate Shapes
        # StandardDQNLoss expects [B, T, Actions] for scalar or [B, T, Actions, Atoms] for categorical
        assert (
            q_preds.ndim >= 3
        ), f"StandardDQNLoss requires at least [B, T, Actions] predictions, got {q_preds.shape}"
        assert (
            actions.ndim == 2
        ), f"StandardDQNLoss requires [B, T] action targets, got {actions.shape}"
        B, T = actions.shape
        num_actions = q_preds.shape[2]

        # 2. Flatten for vectorized action selection
        # [B * T, Actions, ...]
        flat_preds = q_preds.reshape(B * T, num_actions, -1)
        flat_actions = actions.reshape(-1)

        # 3. Select Take Action Predictions: [B * T, ...] (could be scalar or distributions)
        selected_preds = flat_preds[
            torch.arange(B * T, device=self.device), flat_actions
        ]
        if selected_preds.shape[-1] == 1:
            selected_preds = selected_preds.squeeze(-1)

        # 4. Format Targets through the Representation bridge
        target_ingredients = targets
        formatted_target = self.representation.format_target(
            target_ingredients, target_key=self.target_key
        )
        # [B, T, ...] -> [B * T, ...]
        flat_targets = formatted_target.reshape(B * T, -1)
        if flat_targets.shape[-1] == 1:
            flat_targets = flat_targets.squeeze(-1)

        assert (
            selected_preds.shape == flat_targets.shape
        ), f"StandardDQNLoss: shape mismatch {selected_preds.shape} vs {flat_targets.shape}"

        # 5. Apply the actual loss function (CrossEntropy or MSE)
        raw_loss = self.loss_fn(selected_preds, flat_targets, reduction="none")

        # 6. Reshape and Return elementwise [B, T]
        # Sum over categorical atom dimension if present
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)

        return raw_loss.reshape(B, T)

class ChanceQLoss(BaseLoss):
    """Loss for stochastic muzero chance Q heads."""

    def __init__(
        self,
        config: Any,
        device: torch.device,
        representation: Any,
        optimizer_name: str = "default",
        mask_key: str = "afterstate_value_mask",
    ):
        super().__init__(
            config=config,
            device=device,
            pred_key="chance_q_logits",
            target_key="values",
            mask_key=mask_key,
            representation=representation,
            loss_fn=F.cross_entropy,
            optimizer_name=optimizer_name,
            loss_factor=config.chance_q_loss_factor,
        )

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict
    ) -> torch.Tensor:
        """Chance Q computes target value from next step."""
        target_values_next = context.get("target_values_next")
        if target_values_next is None:
            raise KeyError(
                "ChanceQLoss requires 'target_values_next' in context. (TargetBuilder must provide it)"
            )

        # Ingredients for Representation: we overwrite 'values' in local context
        local_ingredients = targets.copy()
        local_ingredients["values"] = target_values_next

        formatted_target = self.representation.format_target(
            local_ingredients, target_key=self.target_key
        )

        pred = predictions[self.pred_key]
        B, T = pred.shape[:2]

        flat_pred = pred.reshape(B * T, -1)
        flat_target = formatted_target.reshape(B * T, -1)

        raw_loss = self.loss_fn(flat_pred, flat_target, reduction="none")
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)

        return self.loss_factor * raw_loss.reshape(B, T)
