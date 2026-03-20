import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseLoss(ABC):
    """
    Unified base class and execution engine for all loss modules.
    Handles data extraction, representation bridging, and masking.
    """

    def __init__(
        self,
        config: Any,
        device: torch.device,
        pred_key: str,
        target_key: str,
        mask_key: str,
        representation: Any,  # Mandatory
        loss_fn: Optional[Any] = None,
        optimizer_name: str = "default",
        loss_factor: float = 1.0,
    ):
        self.config = config
        self.device = device
        self.pred_key = pred_key
        self.target_key = target_key
        self.mask_key = mask_key
        self.representation = representation
        self.loss_fn = loss_fn
        self.optimizer_name = optimizer_name
        self.loss_factor = loss_factor
        self.name = self.__class__.__name__

    @property
    def required_predictions(self) -> set[str]:
        return {self.pred_key}

    @property
    def required_targets(self) -> set[str]:
        return {self.target_key, self.mask_key}

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        """Determine if this loss should be computed for the batch."""
        return True

    def get_mask(self, targets: dict) -> torch.Tensor:
        """Get the mask to apply for this loss (B, T)."""
        mask = targets.get(self.mask_key)
        if mask is None:
            raise KeyError(
                f"Missing required mask '{self.mask_key}' for {self.__class__.__name__}"
            )
        return mask

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """
        Pure Vectorized Execution Engine.
        Returns:
            elementwise_tensor of shape (B, T)
        """
        # 1. Extract [B, T, ...] inputs
        pred = predictions[self.pred_key]
        target_ingredients = targets  # Representation will pull what it needs

        # 2. Format targets through the Representation bridge
        # We pass target_key to help representations that handle multiple inputs
        if hasattr(self.representation, "format_target") and callable(
            self.representation.format_target
        ):
            formatted_target = self.representation.format_target(
                target_ingredients, target_key=self.target_key
            )
        else:
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

        return self.loss_factor * raw_loss.reshape(B, T)

    def compute_priority(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> Optional[torch.Tensor]:
        return None
