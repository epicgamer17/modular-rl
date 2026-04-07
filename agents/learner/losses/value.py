import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from agents.learner.losses.base import BaseLoss


class ValueLoss(BaseLoss):
    """Value prediction loss module (Universal)."""

    def __init__(
        self,
        device: torch.device,
        target_key: str = "values",
        optimizer_name: str = "default",
        mask_key: str = "value_mask",
        loss_fn: Any = F.mse_loss,
        loss_factor: float = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(
            device=device,
            pred_key="values",
            target_key=target_key,
            mask_key=mask_key,
            loss_fn=loss_fn,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
            name=name,
        )


class ClippedValueLoss(BaseLoss):
    """
    PPO Clipped Value Loss.
    Formula: max[(V - V_targ)^2, (clip(V, V_old - eps, V_old + eps) - V_targ)^2]
    """

    def __init__(
        self,
        device: torch.device,
        clip_param: float,
        target_key: str = "returns",
        old_values_key: str = "values",
        optimizer_name: str = "default",
        mask_key: str = "value_mask",
        loss_factor: float = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(
            device=device,
            pred_key="values",
            target_key=target_key,
            mask_key=mask_key,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
            name=name,
        )
        self.clip_param = clip_param
        self.old_values_key = old_values_key

    def compute_loss(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Clipped Value Loss execution.
        Expects:
            predictions['values']: [B, T, 1]
            targets[target_key]: [B, T] or [B, T, 1]
            targets[old_values_key]: [B, T] or [B, T, 1]
        """
        # 1. Extract inputs
        # We assume predictions contains the expected scalar value directly
        values = predictions.get("values_expected", predictions[self.pred_key])
        returns = targets[self.target_key]
        old_values = targets["values"]

        # Ensure shapes match [B, T]
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        if returns.ndim == 3 and returns.shape[-1] == 1:
            returns = returns.squeeze(-1)
        if old_values.ndim == 3 and old_values.shape[-1] == 1:
            old_values = old_values.squeeze(-1)

        # 3. Compute losses
        v_loss_unclipped = (values - returns) ** 2
        v_clipped = old_values + torch.clamp(
            values - old_values, -self.clip_param, self.clip_param
        )
        v_loss_clipped = (v_clipped - returns) ** 2

        # PPO clipped value loss is the maximum of the two
        elementwise_loss = torch.max(v_loss_unclipped, v_loss_clipped)

        # Apply loss factor
        elementwise_loss = self.loss_factor * elementwise_loss

        return elementwise_loss, {}
