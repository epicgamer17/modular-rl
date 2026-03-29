import torch
import torch.nn.functional as F
from typing import Any, Optional, Dict, Tuple
from agents.learner.losses.base import BaseLoss, LossRepresentation

class ValueLoss(BaseLoss):
    """Value prediction loss module (Universal)."""

    def __init__(
        self,
        device: torch.device,
        representation: LossRepresentation,
        target_key: str = "values",
        optimizer_name: str = "default",
        mask_key: str = "value_mask",
        loss_fn: Any = F.mse_loss,
        loss_factor: float = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(
            device=device,
            pred_key="state_value",
            target_key=target_key,
            mask_key=mask_key,
            representation=representation,
            loss_fn=loss_fn,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
            name=name,
        )


class ClippedValueLoss(ValueLoss):
    """PPO Value loss module with optional clipping."""

    def __init__(
        self,
        device: torch.device,
        representation: LossRepresentation,
        clip_param: float,
        target_key: str = "values", # Default changed back to generic
        optimizer_name: str = "default",
        mask_key: str = "value_mask",
        loss_factor: float = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(
            device=device,
            representation=representation,
            target_key=target_key,
            optimizer_name=optimizer_name,
            mask_key=mask_key,
            loss_factor=loss_factor,
            name=name,
        )
        self.clip_param = clip_param

    def compute_loss(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """PPO Value Loss: returns [B, T]"""
        v_pred = predictions[self.pred_key]
        v_old = targets["values"]
        v_target = targets[self.target_key]

        # 1. Capture and Validate [B, T, 1] -> [B, T]
        if v_pred.ndim == 3 and v_pred.shape[-1] == 1:
            v_pred = v_pred.squeeze(-1)
        if v_old.ndim == 3 and v_old.shape[-1] == 1:
            v_old = v_old.squeeze(-1)
        if v_target.ndim == 3 and v_target.shape[-1] == 1:
            v_target = v_target.squeeze(-1)

        assert (
            v_pred.shape == v_old.shape == v_target.shape
        ), f"ClippedValueLoss shape mismatch: pred {v_pred.shape}, old {v_old.shape}, target {v_target.shape}"

        from agents.learner.functional.losses import compute_ppo_value_loss

        loss = compute_ppo_value_loss(
            v_pred, v_old, v_target, self.clip_param
        )

        return self.loss_factor * loss, {}
