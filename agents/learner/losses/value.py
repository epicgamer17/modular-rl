import torch
import torch.nn.functional as F
from typing import Any, Optional
from agents.learner.losses.base import BaseLoss

class ValueLoss(BaseLoss):
    """Value prediction loss module (Universal)."""

    def __init__(
        self,
        config: Any,
        device: torch.device,
        representation: Any,
        target_key: str = "values",
        optimizer_name: str = "default",
        mask_key: str = "value_mask",
        loss_fn: Optional[Any] = None,
        loss_factor: Optional[float] = None,
    ):
        # Fallback to algorithm-specific config or framework defaults
        loss_fn = loss_fn or getattr(config, "value_loss_function", F.mse_loss)
        loss_factor = (
            loss_factor
            if loss_factor is not None
            else getattr(config, "value_loss_factor", 1.0)
        )

        super().__init__(
            config=config,
            device=device,
            pred_key="values",
            target_key=target_key,
            mask_key=mask_key,
            representation=representation,
            loss_fn=loss_fn,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
        )
