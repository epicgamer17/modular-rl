import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional
from learner.pipeline.base import PipelineComponent
from learner.core import Blackboard
from learner.losses.base import apply_infrastructure


class ConsistencyLoss(PipelineComponent):
    """Latent consistency loss (EfficientZero style)."""
    def __init__(
        self,
        loss_factor: float = 1.0,
        mask_key: str = "masks",
        name: str = "consistency_loss",
    ):
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        # Standard implementation: Similarity between predicted latent projections and target projections
        preds = blackboard.predictions.get("projected_latents")
        targets = blackboard.targets.get("consistency_targets")
        
        if preds is None or targets is None:
            return

        B, T = preds.shape[:2]
        
        # Pearson correlation or cosine similarity
        preds_norm = F.normalize(preds, p=2.0, dim=-1)
        targets_norm = F.normalize(targets, p=2.0, dim=-1)
        
        # [B, T]
        elementwise_loss = -(preds_norm * targets_norm).sum(dim=-1) * self.loss_factor

        # Pass through infrastructure
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        # Write out
        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class SigmaLoss(PipelineComponent):
    """Loss for sigma prediction (stochastic MuZero)."""
    def __init__(
        self,
        loss_factor: float = 1.0,
        mask_key: str = "masks",
        name: str = "sigma_loss",
    ):
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions.get("sigma_logits")
        targets = blackboard.targets.get("sigmas")
        
        if preds is None or targets is None:
            return

        B, T = preds.shape[:2]
        raw_loss = F.cross_entropy(preds.flatten(0, 1), targets.flatten(0, 1), reduction="none")
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)
            
        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor

        # Pass through infrastructure
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        # Write out
        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class CommitmentLoss(PipelineComponent):
    """VQ-VAE commitment cost for encoder (stochastic MuZero)."""
    def __init__(
        self,
        loss_factor: float = 1.0,
        mask_key: str = "masks",
        name: str = "commitment_loss",
    ):
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        # Commitment loss usually comes pre-calculated from the world model forward pass
        elementwise_loss = blackboard.predictions.get("commitment_loss")
        if elementwise_loss is None:
            return

        elementwise_loss = elementwise_loss * self.loss_factor

        # Pass through infrastructure
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        # Write out
        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()

