import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional
from learner.pipeline.base import PipelineComponent
from learner.core import Blackboard
from learner.losses.base import apply_infrastructure


class ToPlayLoss(PipelineComponent):
    """Loss for learning to predict whose turn it is."""
    def __init__(
        self,
        loss_factor: float = 1.0,
        mask_key: str = "to_play_mask",
        name: str = "to_play_loss",
    ):
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions["to_plays"]
        targets = blackboard.targets["to_plays"]
        
        B, T = preds.shape[:2]
        
        # Flatten B, T
        raw_loss = F.cross_entropy(preds.flatten(0, 1), targets.flatten(0, 1), reduction="none")
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)
        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor

        # Pass through infrastructure
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        # Write out
        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()
