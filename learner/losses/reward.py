import torch
from typing import Any, Dict, Optional, Tuple
from learner.losses.base import BaseLoss


class RewardLoss(BaseLoss):
    """Reward prediction loss module."""

    def __init__(
        self,
        device: torch.device,
        loss_fn: Any,
        loss_factor: float,
        optimizer_name: str = "default",
        mask_key: str = "reward_mask",
        name: Optional[str] = None,
    ):
        super().__init__(
            device=device,
            pred_key="rewards",
            target_key="rewards",
            mask_key=mask_key,
            loss_fn=loss_fn,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
            name=name,
        )
