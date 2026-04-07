import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from learner.losses.base import BaseLoss


class QBootstrappingLoss(BaseLoss):
    """
    Standard TD target loss for Q-learning.
    Indexes the prediction tensor by the taken actions to compute TD errors.
    """

    def __init__(
        self,
        device: torch.device,
        is_categorical: bool = False,
        loss_fn: Any = None,
        optimizer_name: str = "default",
        mask_key: str = "value_mask",
        name: Optional[str] = None,
    ):
        # 1. Determine Pred/Target keys and default Loss function based on atom_size
        pred_key = "q_logits" if is_categorical else "q_values"
        target_key = "q_logits" if is_categorical else "values"

        if loss_fn is None:
            loss_fn = F.cross_entropy if is_categorical else F.mse_loss

        super().__init__(
            device=device,
            pred_key=pred_key,
            target_key=target_key,
            mask_key=mask_key,
            loss_fn=loss_fn,
            optimizer_name=optimizer_name,
            name=name,
        )

    def compute_loss(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
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

        # 3. Use pre-formatted Targets
        formatted_target = targets[self.target_key]
        flat_targets = formatted_target.reshape(B * T, -1)

        # 4. Final Squeezing and Matching
        # We ensure they are both [B*T] for standard MSE or [B*T, Atoms] for C51
        if selected_preds.ndim > 1 and selected_preds.shape[-1] == 1:
            # Bring B*T, 1 -> B*T
            selected_preds = selected_preds.squeeze(-1)
        if flat_targets.ndim > 1 and flat_targets.shape[-1] == 1:
            # Bring B*T, 1 -> B*T
            flat_targets = flat_targets.squeeze(-1)

        # 5. Apply Loss Function
        if self.pred_key == "q_logits":
            # Multi-atom categorical cross-entropy
            log_probs = F.log_softmax(selected_preds, dim=-1)
            raw_loss = -(flat_targets * log_probs).sum(dim=-1)
        else:
            # Standard scalar regression (MSE)
            raw_loss = self.loss_fn(selected_preds, flat_targets, reduction="none")

        return raw_loss.reshape(B, T), {}


class ChanceQLoss(BaseLoss):
    """Loss for stochastic muzero chance Q heads."""

    def __init__(
        self,
        device: torch.device,
        loss_factor: float,
        optimizer_name: str = "default",
        mask_key: str = "afterstate_value_mask",
        name: Optional[str] = None,
    ):
        super().__init__(
            device=device,
            pred_key="chance_q_logits",
            target_key="chance_values_next",
            mask_key=mask_key,
            loss_fn=F.cross_entropy,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
            name=name,
        )

    def compute_loss(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Chance Q computes target value from next step."""
        chance_values_next = targets.get("chance_values_next")
        if chance_values_next is None:
            raise KeyError(
                "ChanceQLoss requires 'chance_values_next' in targets. (MuZero target builders must provide it)"
            )

        # Ingredients for Representation: chance_values_next is already provided and whitelisted
        formatted_target = targets["chance_values_next"]

        pred = predictions[self.pred_key]
        B, T = pred.shape[:2]

        flat_pred = pred.reshape(B * T, -1)
        flat_target = formatted_target.reshape(B * T, -1)

        raw_loss = self.loss_fn(flat_pred, flat_target, reduction="none")
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)

        return self.loss_factor * raw_loss.reshape(B, T), {}
