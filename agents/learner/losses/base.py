import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, runtime_checkable, Tuple


@runtime_checkable
class LossRepresentation(Protocol):
    """
    Contract for mathematical representations of values (scalars, distributions, etc.).
    """

    @property
    def num_features(self) -> int: ...
    def to_inference(self, logits: torch.Tensor) -> Any: ...
    def to_expected_value(self, logits: torch.Tensor) -> torch.Tensor: ...
    def to_representation(self, scalar_targets: torch.Tensor) -> torch.Tensor: ...
    def format_target(
        self, targets: Dict[str, torch.Tensor], target_key: str = "values"
    ) -> torch.Tensor: ...


@runtime_checkable
class DistributionalRepresentation(LossRepresentation, Protocol):
    """
    Contract for representations that support Bellman projections onto a fixed grid.
    """

    def project_onto_grid(
        self, shifted_support: torch.Tensor, probabilities: torch.Tensor
    ) -> torch.Tensor: ...


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
        representation: LossRepresentation,  # Mandatory
        loss_fn: Optional[Any] = None,
        optimizer_name: str = "default",
        loss_factor: float = 1.0,
        name: Optional[str] = None,
    ):
        self.device = device
        self.pred_key = pred_key
        self.target_key = target_key
        self.mask_key = mask_key
        self.representation = representation
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

        # 2. Format targets through the Representation bridge
        # Target ingredients are pulled directly from the targets dict!
        formatted_target = self.representation.format_target(
            targets, target_key=self.target_key
        )

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

        flat_mask = self.get_mask(targets).reshape(B * T).bool()
        # Performance optimization: ensure targets are probability distributions before computing loss
        # Only applicable if we have multiple features (e.g. atoms/bins) and not just a single scalar
        if flat_target.shape[-1] > 1:
            # We use a broad check to catch mass loss in Bellman projections
            t_sum = flat_target.sum(dim=-1)
            # NOTE: Old MuZero parity testing only. Legacy replay can emit
            # zero-mass categorical targets on masked-valid rows (notably
            # to_play after terminal), and the old learner simply ignored them.
            massful_mask = flat_mask & (t_sum > 0)
            # Filter the sums to ONLY check valid states according to the mask
            valid_t_sum = t_sum[massful_mask]

            # Only run the assertion if there are valid targets in the batch
            if valid_t_sum.numel() > 0:
                is_correct = torch.isclose(valid_t_sum, torch.ones_like(valid_t_sum), atol=1e-3)
                if not torch.all(is_correct):
                    # 1. Identify failing indices
                    invalid_indices = (~is_correct).nonzero(as_tuple=True)[0]
                    # Map back to the original flattened index
                    full_indices = massful_mask.nonzero(as_tuple=True)[0][invalid_indices]
                    
                    # 2. Extract offending data
                    offending_targets = flat_target[full_indices]
                    offending_sums = valid_t_sum[invalid_indices]
                    
                    # 3. Format detailed error report
                    msg = (
                        f"\n[CRITICAL] {self.__class__.__name__}: Categorical targets MUST sum to 1.0 (Probability Mass Loss detected).\n"
                        f"Found {len(invalid_indices)} invalid rows out of {len(valid_t_sum)} valid training samples.\n"
                        f"Mean Sum: {valid_t_sum.mean().item():.4f}\n"
                        f"Example Violations (showing up to 5):\n"
                    )
                    
                    for i in range(min(5, len(invalid_indices))):
                        idx = full_indices[i].item()
                        b_idx, t_idx = idx // T, idx % T
                        s = offending_sums[i].item()
                        t_vals = offending_targets[i].detach().cpu().tolist()
                        msg += f"  -> [Batch {b_idx}, Step {t_idx}] Sum: {s:.6f} | Target: {t_vals}\n"
                    
                    raise AssertionError(msg)
        raw_loss = self.loss_fn(flat_pred, flat_target, reduction="none")

        # 4. Collapse and Reshape to [B, T] result
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)

        elementwise_loss = self.loss_factor * raw_loss.reshape(B, T)

        # Base losses return no extra metrics by default
        return elementwise_loss, {}

    def compute_priority(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        return None
