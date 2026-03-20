from typing import Any
import torch
from agents.learner.losses.base import BaseLoss

class RewardLoss(BaseLoss):
    """Reward prediction loss module."""

    def __init__(
        self,
        config: Any,
        device: torch.device,
        representation: Any,
        optimizer_name: str = "default",
        mask_key: str = "reward_mask",
    ):
        super().__init__(
            config=config,
            device=device,
            pred_key="rewards",
            target_key="rewards",
            mask_key=mask_key,
            representation=representation,
            loss_fn=config.reward_loss_function,
            optimizer_name=optimizer_name,
            loss_factor=config.reward_loss_factor,
        )
