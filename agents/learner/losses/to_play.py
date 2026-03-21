import torch
import torch.nn.functional as F
from typing import Any, Optional
from agents.learner.losses.base import BaseLoss, LossRepresentation

class ToPlayLoss(BaseLoss):
    """Loss for learning to predict who's turn it is."""

    def __init__(
        self,
        device: torch.device,
        representation: LossRepresentation,
        loss_factor: float,
        optimizer_name: str = "default",
        mask_key: str = "to_play_mask",
        name: Optional[str] = None,
    ):
        super().__init__(
            device=device,
            pred_key="to_play_logits",
            target_key="to_plays",
            mask_key=mask_key,
            representation=representation,
            loss_fn=F.cross_entropy,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
            name=name,
        )

class RelativeToPlayLoss(BaseLoss):
    """
    Loss for relative player turn prediction.
    Predicted [B, T, num_players] is a softmax over players relative to root.
    Target logic shifts based on sequence current player.
    """

    def __init__(
        self,
        device: torch.device,
        representation: LossRepresentation,
        loss_factor: float,
        optimizer_name: str = "default",
        mask_key: str = "to_play_mask",
        name: Optional[str] = None,
    ):
        super().__init__(
            device=device,
            pred_key="to_play_logits",
            target_key="to_plays",
            mask_key=mask_key,
            representation=representation,
            loss_fn=F.cross_entropy,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
            name=name,
        )
