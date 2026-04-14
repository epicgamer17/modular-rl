import torch
import torch.nn.functional as F
from typing import Any, Set, Dict
from core import PipelineComponent
from core import Blackboard
from core.path_resolver import resolve_blackboard_path
from core.contracts import Key, ShapeContract, ValueEstimate, ValueTarget, LossScalar, Metric, Scalar
from .infrastructure import apply_infrastructure
from core.validation import assert_same_batch, assert_compatible_value


class ValueLoss(PipelineComponent):
    """
    Standard Value prediction loss component.
    Reads 'values' from predictions and targets.
    """

    def __init__(
        self,
        target_key: str = "values",
        mask_key: str = "value_mask",
        loss_fn: Any = F.mse_loss,
        loss_factor: float = 1.0,
        name: str = "value_loss",
    ):
        self.target_key = target_key
        self.mask_key = mask_key
        self.loss_fn = loss_fn
        self.loss_factor = loss_factor
        self.name = name

        # Deterministic contracts computed at initialization
        self._requires = {
            Key("predictions.values", ValueEstimate[Scalar],
                shape=ShapeContract(has_time=True)),
            Key(self.target_key, ValueTarget[Scalar]),
        }
        self._provides = {
            Key(f"losses.{self.name}", LossScalar): "new",
            Key(f"meta.{self.name}", Metric): "new",
            Key(f"meta.elementwise_losses.{self.name}", Metric): "new",
        }

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    @property
    def constraints(self) -> list[str]:
        return [
            f"same_batch(predictions.values, {self.target_key})",
            "time_aligned(predictions.values, targets)",
        ]

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures both prediction and target exist, are tensors, and aligned."""
        from core.validation import (
            assert_in_blackboard,
            assert_is_tensor,
            assert_same_batch,
            assert_compatible_value,
            assert_shape_sanity,
        )

        assert_in_blackboard(blackboard, "predictions.values")
        assert_in_blackboard(blackboard, self.target_key)

        preds = blackboard.predictions["values"]
        targets = resolve_blackboard_path(blackboard, self.target_key)

        assert_is_tensor(preds, msg=f"in {self.name} (predictions)")
        assert_is_tensor(targets, msg=f"in {self.name} (targets)")

        assert_same_batch(preds, targets, msg=f"in {self.name}")

        # Rigorous shape checks
        assert_shape_sanity(
            preds, min_ndim=3, msg=f"Prediction {self.name} must have [B, T, 1]"
        )

        # Hypothetical alignment for validation purposes
        check_targets = targets
        if check_targets.ndim == preds.ndim - 1:
            check_targets = check_targets.unsqueeze(1)  # Add T=1

        if check_targets.ndim == preds.ndim - 1:
            check_targets = check_targets.unsqueeze(-1)  # Add feature dim

        assert check_targets.shape == preds.shape, (
            f"ValueLoss Contract Violation: targets {targets.shape} (locally aligned to {check_targets.shape}) "
            f"must match predictions {preds.shape}. Expected [B, T, 1]."
        )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """Compute value loss and write to losses."""
        preds = blackboard.predictions["values"]
        targets = resolve_blackboard_path(blackboard, self.target_key)
        # TODO: enforce always time dimension some how.
        # 1. Robust Time Alignment: Handle [B, 1] -> [B, 1, 1] or [B] -> [B, 1, 1]
        if targets.ndim == preds.ndim - 1:
            targets = targets.unsqueeze(1)

        if targets.ndim == preds.ndim - 1:
            targets = targets.unsqueeze(-1)

        # 2. Flatten leads dimensions [B, T] into [N] for standard PyTorch loss functions
        B, T = preds.shape[:2]
        p_flat = preds.flatten(0, 1)
        t_flat = targets.flatten(0, 1)

        raw_loss = self.loss_fn(p_flat, t_flat, reduction="none")

        # Reshape to [B, T]
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)

        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor

        # Pass through infrastructure
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        # Write out
        return {
            f"losses.{self.name}": scalar_loss,
            f"meta.{self.name}": scalar_loss.item(),
            f"meta.elementwise_losses.{self.name}": elementwise_loss,
        }


class ClippedValueLoss(PipelineComponent):
    """
    PPO Clipped Value Loss.
    Formula: max[(V - V_targ)^2, (clip(V, V_old - eps, V_old + eps) - V_targ)^2]
    """

    def __init__(
        self,
        clip_param: float,
        target_key: str = "returns",
        old_values_key: str = "values",
        mask_key: str = "value_mask",
        loss_factor: float = 1.0,
        name: str = "value_loss",
    ):
        self.clip_param = clip_param
        self.target_key = target_key
        self.old_values_key = old_values_key
        self.mask_key = mask_key
        self.loss_factor = loss_factor
        self.name = name

        # Deterministic contracts computed at initialization
        self._requires = {
            Key("predictions.values", ValueEstimate[Scalar]),
            Key(self.target_key, ValueTarget[Scalar]),
            Key(self.old_values_key, ValueEstimate[Scalar]),
        }
        self._provides = {
            Key(f"losses.{self.name}", LossScalar): "new",
            Key(f"meta.{self.name}", Metric): "new",
            Key(f"meta.elementwise_losses.{self.name}", Metric): "new",
        }

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    @property
    def constraints(self) -> list[str]:
        return ["clipped_surrogate_invariant(returns, values, old_values)"]

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures all 3 required tensors exist and have matching shapes."""
        from core.validation import (
            assert_in_blackboard,
            assert_is_tensor,
            assert_same_batch,
            assert_shape_sanity,
        )

        # Check existence
        assert_in_blackboard(blackboard, "predictions.values")
        assert_in_blackboard(blackboard, self.target_key)
        assert_in_blackboard(blackboard, self.old_values_key)

        # Extract for shape/type checks
        values = blackboard.predictions.get(
            "values_expected", blackboard.predictions["values"]
        )
        returns = resolve_blackboard_path(blackboard, self.target_key)
        old_values = resolve_blackboard_path(blackboard, self.old_values_key)

        # Assertions
        assert_is_tensor(values, msg=f"in {self.name} (values)")
        assert_is_tensor(returns, msg=f"in {self.name} (returns)")
        assert_is_tensor(old_values, msg=f"in {self.name} (old_values)")

        assert_same_batch(values, returns, msg=f"in {self.name}")
        assert_same_batch(values, old_values, msg=f"in {self.name}")

        # PPO requires [B, T] or identical shapes
        assert_shape_sanity(values, min_ndim=1, max_ndim=3, msg=f"for {self.name}")
        assert (
            values.shape == returns.shape == old_values.shape
        ), f"Shape mismatch in {self.name}: {values.shape} vs {returns.shape} vs {old_values.shape}"

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        # 1. Extract inputs
        values = blackboard.predictions.get(
            "values_expected", blackboard.predictions["values"]
        )
        returns = resolve_blackboard_path(blackboard, self.target_key)
        old_values = resolve_blackboard_path(blackboard, self.old_values_key)

        # Ensure shapes match [B, T]
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        if returns.ndim == 3 and returns.shape[-1] == 1:
            returns = returns.squeeze(-1)
        if old_values.ndim == 3 and old_values.shape[-1] == 1:
            old_values = old_values.squeeze(-1)

        # 3. Compute losses
        v_loss_unclipped = (values - returns) ** 2
        v_clipped = old_values + torch.clamp(
            values - old_values, -self.clip_param, self.clip_param
        )
        v_loss_clipped = (v_clipped - returns) ** 2

        # PPO clipped value loss is the maximum of the two
        elementwise_loss = torch.max(v_loss_unclipped, v_loss_clipped)
        elementwise_loss = elementwise_loss * self.loss_factor

        # Pass through infrastructure
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        # Write out
        return {
            f"losses.{self.name}": scalar_loss,
            f"meta.{self.name}": scalar_loss.item(),
            f"meta.elementwise_losses.{self.name}": elementwise_loss,
        }
