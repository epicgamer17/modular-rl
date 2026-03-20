import torch
import torch.nn.functional as F
from typing import Any, Optional
from agents.learner.losses.base import BaseLoss

class ToPlayLoss(BaseLoss):
    """Loss for learning to predict who's turn it is."""

    def __init__(
        self,
        config: Any,
        device: torch.device,
        representation: Any,
        optimizer_name: str = "default",
        mask_key: str = "to_play_mask",
    ):
        super().__init__(
            config=config,
            device=device,
            pred_key="to_plays",
            target_key="to_plays",
            mask_key=mask_key,
            representation=representation,
            loss_fn=F.cross_entropy,
            optimizer_name=optimizer_name,
            loss_factor=config.to_play_loss_factor,
        )

class RelativeToPlayLoss(BaseLoss):
    """
    Loss for relative player turn prediction.
    Predicted [B, T, num_players] is a softmax over players relative to root.
    Target logic shifts based on sequence current player.
    """

    def __init__(
        self,
        config: Any,
        device: torch.device,
        representation: Any,
        optimizer_name: str = "default",
        mask_key: str = "to_play_mask",
    ):
        super().__init__(
            config=config,
            device=device,
            pred_key="to_plays",
            target_key="to_plays",
            mask_key=mask_key,
            representation=representation,
            loss_fn=F.cross_entropy,
            optimizer_name=optimizer_name,
            loss_factor=config.to_play_loss_factor,
        )
