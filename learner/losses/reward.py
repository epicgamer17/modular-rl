import torch
from typing import Any, Dict, Optional
from learner.pipeline.base import PipelineComponent
from learner.core import Blackboard
from learner.losses.base import apply_infrastructure


class RewardLoss(PipelineComponent):
    """Reward prediction loss module."""
    def __init__(
        self,
        loss_fn: Any,
        loss_factor: float = 1.0,
        mask_key: str = "reward_mask",
        name: str = "reward_loss",
    ):
        self.loss_fn = loss_fn
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions["rewards"]
        targets = blackboard.targets["rewards"]
        
        B, T = preds.shape[:2]
        
        # Flatten B, T
        raw_loss = self.loss_fn(preds.flatten(0, 1), targets.flatten(0, 1), reduction="none")
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)
        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor

        # Pass through infrastructure
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        # Write out
        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()
