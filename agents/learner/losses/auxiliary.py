import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from agents.learner.losses.base import BaseLoss, LossRepresentation


class ConsistencyLoss(BaseLoss):
    """Latent consistency loss (stochastic/standard MuZero)."""

    def __init__(
        self,
        device: torch.device,
        representation: LossRepresentation,
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
            representation=representation,
            loss_fn=F.mse_loss,  # TODO: This should be cosine similarity
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
            name=name,
        )


class SigmaLoss(BaseLoss):
    """Loss for sigma prediction (stochastic MuZero)."""

    def __init__(
        self,
        device: torch.device,
        representation: LossRepresentation,
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
            representation=representation,
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
        representation: LossRepresentation,
        loss_factor: float = 1.0,  # Ensure loss_factor is explicitly accepted
        optimizer_name: str = "default",
        mask_key: str = "commitment_mask",
        name: Optional[str] = None,
    ):
        super().__init__(
            device=device,
            pred_key="commitment_loss",
            target_key="dummy",  # Safely overridden below
            mask_key=mask_key,
            representation=representation,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
            name=name,
        )

    @property
    def required_targets(self) -> set[str]:
        # Commitment loss is internal to the network latents; it requires no external targets.
        return {self.mask_key}

    def compute_loss(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        raw_loss = predictions[self.pred_key]
        mask = self.get_mask(targets)

        # 1. Guarantee the Universal [B, T] contract regardless of network output
        if raw_loss.ndim == 0:
            # Network returned a global scalar
            raw_loss = raw_loss.expand(mask.shape)
        elif raw_loss.ndim == 1:
            # Network returned [B]
            raw_loss = raw_loss.unsqueeze(-1).expand(mask.shape)

        # 2. Apply the scaling factor and return
        return self.loss_factor * raw_loss, {}
