import torch
from typing import Any, Set
from core import PipelineComponent, Blackboard
from core.contracts import Key, SemanticType, PolicyLogits, ValueTarget, ValueEstimate


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

    @property
    def requires(self) -> Set[Key]:
        return {Key("meta.elementwise_losses", SemanticType)}

    @property
    def provides(self) -> Set[Key]:
        return {Key("meta.priorities", SemanticType)}

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
        """Extracts priorities and stores them in blackboard.meta."""
        elementwise_losses = blackboard.meta.get("elementwise_losses", {})
        if self.loss_key in elementwise_losses:
            loss = elementwise_losses[self.loss_key]
            # [B, T] -> [B]
            if loss.ndim == 1:
                blackboard.meta['priorities'] = loss.detach()
            elif self.reduction == "root":
                blackboard.meta['priorities'] = loss[:, 0].detach()
            else:
                blackboard.meta['priorities'] = loss.max(dim=1).values.detach()


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
        target_key: str = "values",
        pred_key: str = "values",
    ):
        self.value_representation = value_representation
        self.target_key = target_key
        self.pred_key = pred_key

    @property
    def requires(self) -> Set[Key]:
        return {
            Key(f"predictions.{self.pred_key}", ValueEstimate),
            Key(f"targets.{self.target_key}", ValueTarget)
        }

    @property
    def provides(self) -> Set[Key]:
        return {Key("meta.priorities", SemanticType)}

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
        """Computes MSE of expected value error at root and stores in blackboard.meta."""
        if self.pred_key not in blackboard.predictions or self.target_key not in blackboard.targets:
            return

        # 1. Predictions: Distribution -> Expected Scalar Value [B, T]
        pred_logits = blackboard.predictions[self.pred_key]
        pred_scalars = self.value_representation.to_expected_value(pred_logits)

        # 2. Targets: Raw Scalar Value [B, T]
        target_scalars = blackboard.targets[self.target_key]

        # 3. Compute root TD-error (MSE)
        # Ensure target_scalars is [B, T] by squeezing if it's [B, T, 1]
        if target_scalars.ndim == 3 and target_scalars.shape[-1] == 1:
            target_scalars = target_scalars.squeeze(-1)
            
        root_pred = pred_scalars[:, 0]
        root_target = target_scalars[:, 0]

        assert root_target.shape == root_pred.shape, (
            f"ExpectedValueErrorPriorityComponent: Shape mismatch between "
            f"root_target {root_target.shape} and root_pred {root_pred.shape}"
        )

        error = (root_target - root_pred) ** 2
        blackboard.meta['priorities'] = error.detach()
