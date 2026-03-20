import torch
import torch.nn.functional as F
from typing import Any, Optional
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
            pred_key="values",
            target_key=target_key,
            mask_key=mask_key,
            representation=representation,
            loss_fn=loss_fn,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
            name=name,
        )
