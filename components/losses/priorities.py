from typing import Any, Set, Dict
import torch
from core import PipelineComponent, Blackboard
from core.path_resolver import resolve_blackboard_path
from core.contracts import (
    Key,
    SemanticType,
    Policy,
    ValueTarget,
    ValueEstimate,
    Priority,
    Metric,
)


class LossPriorityComponent(PipelineComponent):
    """
    Extracts priorities from elementwise losses stored in blackboard.meta["elementwise_losses"].

    Reduction modes:
    - "root": Uses loss at the root step (t=0). Standard for MuZero.
    - "max": Uses the maximum loss over the entire sequence. Standard for DQN/Rainbow.
    """

    def __init__(self, loss_key: str = "q_loss", reduction: str = "max"):
        self.loss_key = loss_key
        assert reduction in ("root", "max"), f"Unknown reduction: {reduction}"
        self.reduction = reduction

        # Deterministic contracts computed at initialization
        self._requires = {Key(f"meta.elementwise_losses.{self.loss_key}", Metric)}
        self._provides = {Key("meta.priorities", Priority): "new"}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures elementwise_losses dict exists and contains the target loss key."""
        elementwise_losses = blackboard.meta["elementwise_losses"]
        loss = elementwise_losses[self.loss_key]
        assert torch.is_tensor(
            loss
        ), f"Expected tensor for elementwise loss '{self.loss_key}', got {type(loss)}"

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """Extracts priorities and stores them in blackboard.meta."""
        elementwise_losses = blackboard.meta["elementwise_losses"]
        loss = elementwise_losses[self.loss_key]
        # [B, T] -> [B]
        if loss.ndim == 1:
            return {"meta.priorities": loss.detach()}
        elif self.reduction == "root":
            return {"meta.priorities": loss[:, 0].detach()}
        else:
            return {"meta.priorities": loss.max(dim=1).values.detach()}
        return {}


class ExpectedValueErrorPriorityComponent(PipelineComponent):
    """
    MuZero Standard for Distributional RL: Priority is based on the MSE
    of the expected value error at the root (t=0) - NOT the cross-entropy loss.
    This prevents priorities from being skewed by different math scales.
    [B, T, bins] -> [B, T] -> [B]
    """

    def __init__(
        self,
        value_representation: Any,
        prediction_key: str = "predictions.values",
        target_key: str = "targets.values",
        dest_key: str = "meta.priorities",
    ):
        self.value_representation = value_representation
        self.prediction_key = prediction_key
        self.target_key = target_key
        self.dest_key = dest_key

        # Deterministic contracts computed at initialization
        self._requires = {
            Key(self.prediction_key, ValueEstimate),
            Key(self.target_key, ValueTarget),
        }
        self._provides = {Key(self.dest_key, Priority): "new"}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures prediction and target keys exist and are tensors with time dimension."""
        from core.validation import (
            assert_in_blackboard,
            assert_is_tensor,
            assert_shape_sanity,
        )

        try:
            pred_logits = resolve_blackboard_path(blackboard, self.prediction_key)
            target_scalars = resolve_blackboard_path(blackboard, self.target_key)
        except (KeyError, AttributeError):
            return  # Keys may not exist yet; execute() handles gracefully

        assert_is_tensor(
            pred_logits, msg=f"in ExpectedValueErrorPriorityComponent (predictions)"
        )
        assert_is_tensor(
            target_scalars, msg=f"in ExpectedValueErrorPriorityComponent (targets)"
        )
        assert (
            pred_logits.ndim >= 2
        ), f"Predictions must have at least [B, T], got {pred_logits.shape}"
        assert (
            target_scalars.ndim >= 1
        ), f"Targets must have at least [B], got {target_scalars.shape}"

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """Computes MSE of expected value error at root and stores in blackboard.meta."""
        try:
            pred_logits = resolve_blackboard_path(blackboard, self.prediction_key)
            target_scalars = resolve_blackboard_path(blackboard, self.target_key)
        except (KeyError, AttributeError):
            return {}

        # 1. Predictions: Distribution -> Expected Scalar Value [B, T]
        pred_scalars = self.value_representation.to_expected_value(pred_logits)

        # 2. Targets: Raw Scalar Value [B, T]
        # Ensure target_scalars is [B, T] by squeezing if it's [B, T, 1]
        if target_scalars.ndim == 3 and target_scalars.shape[-1] == 1:
            target_scalars = target_scalars.squeeze(-1)

        root_pred = pred_scalars[:, 0]
        root_target = target_scalars[:, 0]

        error = (root_target - root_pred) ** 2
        return {"meta.priorities": error.detach()}
