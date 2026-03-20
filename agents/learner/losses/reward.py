import torch
from typing import Any
from agents.learner.losses.base import BaseLoss, LossRepresentation

class RewardLoss(BaseLoss):
    """Reward prediction loss module."""

    def __init__(
        self,
        device: torch.device,
        representation: LossRepresentation,
        loss_fn: Any,
        loss_factor: float,
        optimizer_name: str = "default",
        mask_key: str = "reward_mask",
    ):
        super().__init__(
            device=device,
            pred_key="rewards",
            target_key="rewards",
            mask_key=mask_key,
            representation=representation,
            loss_fn=loss_fn,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
        )
