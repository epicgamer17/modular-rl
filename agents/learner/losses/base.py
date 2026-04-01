import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseLoss(ABC):
    """
    Unified base class and execution engine for all loss modules.
    Handles data extraction, representation bridging, and masking.
    """

    def __init__(
        self,
        device: torch.device,
        pred_key: str,
        target_key: str,
        mask_key: str,
        loss_fn: Optional[Any] = None,
        optimizer_name: str = "default",
        loss_factor: float = 1.0,
        name: Optional[str] = None,
    ):
        self.device = device
        self.pred_key = pred_key
        self.target_key = target_key
        self.mask_key = mask_key
        self.loss_fn = loss_fn
        self.optimizer_name = optimizer_name
        self.loss_factor = loss_factor
        self.name = name or self.__class__.__name__

    @property
    def required_predictions(self) -> set[str]:
        return {self.pred_key}

    @property
    def required_targets(self) -> set[str]:
        return {self.target_key, self.mask_key}

    def get_mask(self, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get the mask to apply for this loss (B, T)."""
        mask = targets.get(self.mask_key)
        if mask is None:
            raise KeyError(
                f"Missing required mask '{self.mask_key}' for {self.__class__.__name__}"
            )
        return mask

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Pure Vectorized Execution Engine.
        Returns:
            elementwise_tensor: [B, T]
            metrics: Dictionary of auxiliary logging metrics
        """
        # 1. Extract [B, T, ...] inputs
        pred = predictions[self.pred_key]

        # 2. Extract targets [B, T, ...]
        formatted_target = targets[self.target_key]

        # 3. Apply raw PyTorch loss function
        assert pred.shape == formatted_target.shape, (
            f"{self.__class__.__name__}: shape mismatch between pred {pred.shape} "
            f"and formatted_target {formatted_target.shape}"
        )

        # Determine B, T directly from the primary prediction tensor
        B, T = pred.shape[:2]

        # Flatten B, T to handle universal T contract correctly
        orig_shape = pred.shape
        flat_pred = pred.reshape(B * T, *orig_shape[2:])
        flat_target = formatted_target.reshape(B * T, *orig_shape[2:])

        raw_loss = self.loss_fn(flat_pred, flat_target, reduction="none")

        # 4. Collapse and Reshape to [B, T] result
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)

        elementwise_loss = self.loss_factor * raw_loss.reshape(B, T)

        # Base losses return no extra metrics by default
        return elementwise_loss, {}
