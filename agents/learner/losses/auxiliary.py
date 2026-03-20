import torch
import torch.nn.functional as F
from typing import Any
from agents.learner.losses.base import BaseLoss

class ConsistencyLoss(BaseLoss):
    """Latent consistency loss (stochastic/standard MuZero)."""

    def __init__(
        self,
        device: torch.device,
        representation: Any,
        agent_network: Any,
        loss_factor: float,
        optimizer_name: str = "default",
        mask_key: str = "consistency_mask",
    ):
        super().__init__(
            device=device,
            pred_key="consistency_logits",
            target_key="targets_latent",
            mask_key=mask_key,
            representation=representation,
            loss_fn=F.cross_entropy,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
        )

class SigmaLoss(BaseLoss):
    """Loss for sigma prediction (stochastic MuZero)."""

    def __init__(
        self,
        device: torch.device,
        representation: Any,
        loss_factor: float,
        optimizer_name: str = "default",
        mask_key: str = "sigma_mask",
    ):
        super().__init__(
            device=device,
            pred_key="sigma_logits",
            target_key="sigmas",
            mask_key=mask_key,
            representation=representation,
            loss_fn=F.cross_entropy,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
        )

class CommitmentLoss(BaseLoss):
    """VQ-VAE commitment cost for encoder (stochastic MuZero)."""

    def __init__(
        self,
        device: torch.device,
        representation: Any,
        optimizer_name: str = "default",
        mask_key: str = "commitment_mask",
    ):
        super().__init__(
            device=device,
            pred_key="commitment_loss",
            target_key="commitment_loss",
            mask_key=mask_key,
            representation=representation,
            optimizer_name=optimizer_name,
        )

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict
    ) -> torch.Tensor:
        """Scalar-to-vectorized: returns [B, T] of loss values"""
        B, T = predictions["commitment_loss"].shape[:2]
        # Commitment loss is usually already [B, T] from the VQ head
        return predictions["commitment_loss"]
