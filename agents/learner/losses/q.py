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
        device: torch.device,
        representation: Any,
        is_categorical: bool = False,
        loss_fn: Any = None,
        optimizer_name: str = "default",
        mask_key: str = "value_mask",
    ):
        # 1. Determine Pred/Target keys and default Loss function based on atom_size
        pred_key = "q_logits" if is_categorical else "q_values"
        target_key = "q_logits" if is_categorical else "q_values"

        if loss_fn is None:
            loss_fn = F.cross_entropy if is_categorical else F.mse_loss

        super().__init__(
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
        assert (
            q_preds.ndim >= 3
        ), f"QBootstrappingLoss requires at least [B, T, Actions] predictions, got {q_preds.shape}"
        B, T = actions.shape
        num_actions = q_preds.shape[2]

        # 2. Select Take Action Predictions: [B, T, Atoms] or [B, T, 1]
        flat_preds = q_preds.reshape(B * T, num_actions, -1)
        flat_actions = actions.reshape(-1)
        selected_preds = flat_preds[
            torch.arange(B * T, device=self.device), flat_actions
        ]

        # 3. Format Targets through the Representation bridge
        formatted_target = self.representation.format_target(
            targets, target_key=self.target_key
        )
        flat_targets = formatted_target.reshape(B * T, -1)

        # 4. Final Squeezing and Matching
        if selected_preds.shape[-1] == 1:
            selected_preds = selected_preds.squeeze(-1)
        if flat_targets.shape[-1] == 1:
            flat_targets = flat_targets.squeeze(-1)

        # 5. Apply Loss Function
        if selected_preds.shape[-1] > 1:
            # Multi-atom categorical cross-entropy
            log_probs = F.log_softmax(selected_preds, dim=-1)
            raw_loss = -(flat_targets * log_probs).sum(dim=-1)
        else:
            # Standard scalar regression (MSE)
            selected_preds = selected_preds.squeeze(-1)
            flat_targets = flat_targets.squeeze(-1)
            raw_loss = self.loss_fn(selected_preds, flat_targets, reduction="none")

        return raw_loss.reshape(B, T)

class ChanceQLoss(BaseLoss):
    """Loss for stochastic muzero chance Q heads."""

    def __init__(
        self,
        device: torch.device,
        representation: Any,
        loss_factor: float,
        optimizer_name: str = "default",
        mask_key: str = "afterstate_value_mask",
    ):
        super().__init__(
            device=device,
            pred_key="chance_q_logits",
            target_key="values",
            mask_key=mask_key,
            representation=representation,
            loss_fn=F.cross_entropy,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
        )

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict
    ) -> torch.Tensor:
        """Chance Q computes target value from next step."""
        chance_values_next = targets.get("chance_values_next")
        if chance_values_next is None:
            raise KeyError(
                "ChanceQLoss requires 'chance_values_next' in targets. (MuZeroTargetBuilder must provide it)"
            )

        # Ingredients for Representation: we overwrite 'values' in local context
        local_ingredients = targets.copy()
        local_ingredients["values"] = chance_values_next

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
