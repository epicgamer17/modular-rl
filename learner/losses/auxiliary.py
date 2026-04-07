import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from learner.losses.base import BaseLoss


class ConsistencyLoss(BaseLoss):
    """Latent consistency loss (stochastic/standard MuZero)."""

    def __init__(
        self,
        device: torch.device,
        agent_network: Any,
        loss_factor: float,
        optimizer_name: str = "default",
        mask_key: str = "consistency_mask",
        name: Optional[str] = None,
    ):
        super().__init__(
            device=device,
            pred_key="consistency_logits",
            target_key="targets_latent",
            mask_key=mask_key,
            loss_fn=F.cross_entropy,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
            name=name,
        )


class SigmaLoss(BaseLoss):
    """Loss for sigma prediction (stochastic MuZero)."""

    def __init__(
        self,
        device: torch.device,
        loss_factor: float,
        optimizer_name: str = "default",
        mask_key: str = "sigma_mask",
        name: Optional[str] = None,
    ):
        super().__init__(
            device=device,
            pred_key="sigma_logits",
            target_key="sigmas",
            mask_key=mask_key,
            loss_fn=F.cross_entropy,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
            name=name,
        )


class CommitmentLoss(BaseLoss):
    """VQ-VAE commitment cost for encoder (stochastic MuZero)."""

    def __init__(
        self,
        device: torch.device,
        optimizer_name: str = "default",
        mask_key: str = "commitment_mask",
        name: Optional[str] = None,
    ):
        super().__init__(
            device=device,
            pred_key="commitment_loss",
            target_key="commitment_loss",
            mask_key=mask_key,
            optimizer_name=optimizer_name,
            name=name,
        )

    def compute_loss(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Scalar-to-vectorized: returns [B, T] of loss values"""
        return predictions["commitment_loss"], {}
