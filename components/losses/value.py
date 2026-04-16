import torch
import torch.nn.functional as F
from typing import Any, Set, Dict, Optional, Tuple
from core import PipelineComponent
from core import Blackboard
from core.path_resolver import resolve_blackboard_path
from core.contracts import (
    Key,
    ShapeContract,
    ValueEstimate,
    ValueTarget,
    LossScalar,
    Metric,
    Scalar,
    Logits,
    Probs,
)
from .infrastructure import apply_infrastructure


class ScalarValueLoss(PipelineComponent):
    """
    Scalar value prediction loss (MSE or Huber).

    Predictions: [B, T, 1] scalar value estimates.
    Targets: [B, T, 1] scalar value targets.
    Loss: F.mse_loss or F.huber_loss applied element-wise.
    """

    def __init__(
        self,
        target_key: str = "targets.values",
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

        self._requires = {
            Key(
                "predictions.values",
                ValueEstimate[Scalar],
                shape=ShapeContract(semantic_shape=("B", "T", "A"), event_shape=(1,)),
            ),
            Key(
                self.target_key,
                ValueTarget[Scalar],
                shape=ShapeContract(semantic_shape=("B", "T", "A"), event_shape=(1,)),
            ),
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
        """Ensures both prediction and target exist, are tensors, and shape-aligned."""
        from core.validation import (
            assert_in_blackboard,
            assert_is_tensor,
            assert_same_batch,
            assert_shape_sanity,
        )

        assert_in_blackboard(blackboard, "predictions.values")
        assert_in_blackboard(blackboard, self.target_key)

        preds = blackboard.predictions["values"]
        targets = resolve_blackboard_path(blackboard, self.target_key)

        assert_is_tensor(preds, msg=f"in {self.name} (predictions)")
        assert_is_tensor(targets, msg=f"in {self.name} (targets)")
        assert_same_batch(preds, targets, msg=f"in {self.name}")
        assert_shape_sanity(
            preds, min_rank=3, msg=f"Prediction {self.name} must have [B, T, 1]"
        )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """Compute scalar value loss via MSE/Huber."""
        preds = blackboard.predictions["values"]
        targets = resolve_blackboard_path(blackboard, self.target_key)

        B, T = preds.shape[:2]

        # Align target dims to [B, T, 1]
        if targets.shape != (B, T, 1):
            targets = targets.reshape(B, T, 1)
        p_flat = preds.flatten(0, 1)  # [B*T, 1]
        t_flat = targets.flatten(0, 1)  # [B*T, 1]

        raw_loss = self.loss_fn(p_flat, t_flat, reduction="none")

        # Sanity check: ensure loss is elementwise-compatible
        expected_shape = p_flat.shape
        assert raw_loss.shape == expected_shape, (
            f"{self.name}: shape mismatch between predictions {p_flat.shape} and targets {t_flat.shape}. "
            f"Broadcasting produced output {raw_loss.shape}."
        )

        # Reduce event dim -> [B*T]
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)

        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        return {
            f"losses.{self.name}": scalar_loss,
            f"meta.{self.name}": scalar_loss.item(),
            f"meta.elementwise_losses.{self.name}": elementwise_loss,
        }


class CategoricalValueLoss(PipelineComponent):
    """
    Categorical (distributional) value loss for C51 / MuZero style networks.

    Predictions: [B, T, num_atoms] raw logits over support atoms.
    Targets: [B, T, num_atoms] target probability distribution.
    Loss: cross-entropy  -(target_probs * log_softmax(logits)).sum(dim=-1)
    """

    def __init__(
        self,
        num_atoms: int,
        target_key: str = "targets.values_projected",
        mask_key: str = "value_mask",
        loss_factor: float = 1.0,
        name: str = "value_loss",
    ):
        self.num_atoms = num_atoms
        self.target_key = target_key
        self.mask_key = mask_key
        self.loss_factor = loss_factor
        self.name = name

        self._requires = {
            Key(
                "predictions.values",
                ValueEstimate[Logits],
                shape=ShapeContract(semantic_shape=("B", "T", "A"), event_shape=(num_atoms,)),
            ),
            Key(
                self.target_key,
                ValueTarget[Probs],
                shape=ShapeContract(semantic_shape=("B", "T", "A"), event_shape=(num_atoms,)),
            ),
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
            f"event_shape_match(predictions.values, {self.target_key}, num_atoms={self.num_atoms})",
        ]

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures logits and target probs exist, are tensors, and have matching atom counts."""
        from core.validation import (
            assert_in_blackboard,
            assert_is_tensor,
            assert_same_batch,
            assert_shape_sanity,
        )

        assert_in_blackboard(blackboard, "predictions.values")
        assert_in_blackboard(blackboard, self.target_key)

        logits = blackboard.predictions["values"]
        target_probs = resolve_blackboard_path(blackboard, self.target_key)

        assert_is_tensor(logits, msg=f"in {self.name} (prediction logits)")
        assert_is_tensor(target_probs, msg=f"in {self.name} (target probs)")
        assert_same_batch(logits, target_probs, msg=f"in {self.name}")
        assert_shape_sanity(
            logits,
            min_rank=3,
            msg=f"Prediction {self.name} must have [B, T, num_atoms]",
        )

        assert logits.shape[-1] == self.num_atoms, (
            f"{self.name}: prediction logits last dim {logits.shape[-1]} "
            f"!= expected num_atoms {self.num_atoms}"
        )
        assert target_probs.shape[-1] == self.num_atoms, (
            f"{self.name}: target probs last dim {target_probs.shape[-1]} "
            f"!= expected num_atoms {self.num_atoms}"
        )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """Compute categorical cross-entropy value loss."""
        logits = blackboard.predictions["values"]
        target_probs = resolve_blackboard_path(blackboard, self.target_key)

        B, T = logits.shape[:2]

        # Align target dims to [B, T, num_atoms]
        if target_probs.shape != (B, T, self.num_atoms):
            target_probs = target_probs.reshape(B, T, self.num_atoms)

        # Flatten [B, T] -> [N] for loss computation
        logits_flat = logits.flatten(0, 1)  # [B*T, num_atoms]
        probs_flat = target_probs.flatten(0, 1)  # [B*T, num_atoms]

        # Cross-entropy: -(target_probs * log_softmax(logits)).sum(dim=-1)
        log_probs = F.log_softmax(logits_flat, dim=-1)
        raw_loss = -(probs_flat * log_probs).sum(dim=-1)  # [B*T]

        # Sanity check: ensure loss output rank is correct [B*T]
        assert (
            raw_loss.ndim == 1
        ), f"{self.name}: categorical loss reduction failed. Expected [B*T], got {raw_loss.shape}."

        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

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

        self._requires = {
            Key(
                "predictions.values",
                ValueEstimate[Scalar],
                shape=ShapeContract(
                    semantic_shape=("B", "T", "A"),
                    event_shape=(1,),
                ),
            ),
            Key(
                self.target_key,
                ValueTarget[Scalar],
                shape=ShapeContract(
                    semantic_shape=("B", "T", "A"),
                    event_shape=(1,),
                ),
            ),
            Key(
                self.old_values_key,
                ValueEstimate[Scalar],
                shape=ShapeContract(
                    semantic_shape=("B", "T", "A"),
                    event_shape=(1,),
                ),
            ),
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
        # We allow ndim difference of 1 or 2 as long as they can be aligned to [B, T, 1]
        assert_shape_sanity(values, min_rank=1, max_rank=3, msg=f"for {self.name}")
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

        # 2. Robust Shape Alignment: Ensure buffer data matches [B, T, 1] structure of predictions
        # ModularAgentNetwork.learner_inference returns [B, 1, 1] or [B, T, 1]
        # Buffer usually provides [B] or [B, T]
        B, T = values.shape[:2]

        if values.shape != (B, T, 1):
            values = values.reshape(B, T, 1)
        if returns.shape != (B, T, 1):
            returns = returns.reshape(B, T, 1)
        if old_values.shape != (B, T, 1):
            old_values = old_values.reshape(B, T, 1)

        # 3. Compute losses
        # Sanity check: ensure all inputs are elementwise-compatible [B, T, 1]
        expected_shape = values.shape
        assert returns.shape == expected_shape, (
            f"{self.name}: shape mismatch between values {values.shape} and returns {returns.shape}. "
            "Broadcasting is prohibited for loss targets."
        )
        assert old_values.shape == expected_shape, (
            f"{self.name}: shape mismatch between values {values.shape} and old_values {old_values.shape}. "
            "Broadcasting is prohibited for loss targets."
        )

        v_loss_unclipped = (values - returns) ** 2
        v_clipped = old_values + torch.clamp(
            values - old_values, -self.clip_param, self.clip_param
        )
        v_loss_clipped = (v_clipped - returns) ** 2

        # PPO clipped value loss is the maximum of the two
        elementwise_loss = torch.max(v_loss_unclipped, v_loss_clipped)

        # 4. Final sanity Check: Loss must be elementwise-compatible with predictions
        assert elementwise_loss.shape == expected_shape, (
            f"{self.name}: internal shape error. Elementwise loss {elementwise_loss.shape} "
            f"does not match expected {expected_shape}."
        )
        elementwise_loss = elementwise_loss * self.loss_factor

        # Pass through infrastructure (masking, mean, etc.)
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        # Write out
        return {
            f"losses.{self.name}": scalar_loss,
            f"meta.{self.name}": scalar_loss.item(),
            f"meta.elementwise_losses.{self.name}": elementwise_loss,
        }
