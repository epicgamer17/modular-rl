import torch
from typing import Any
from core import PipelineComponent, Blackboard

class RootTDErrorComponent(PipelineComponent):
    """
    MuZero Standard: Priority is determined by the loss at the root step (t=0).
    Usually uses 'ValueLoss' as the source.
    """
    def __init__(self, loss_key: str = "ValueLoss"):
        self.loss_key = loss_key

    def execute(self, blackboard: Blackboard) -> None:
        """
        Extracts priorities from elementwise losses and stores them in blackboard.meta.
        """
        elementwise_losses = blackboard.meta.get("elementwise_losses", {})
        if self.loss_key in elementwise_losses:
            loss = elementwise_losses[self.loss_key]
            # [B, T] -> [B]
            if loss.ndim == 1:
                blackboard.meta['priorities'] = loss.detach()
            else:
                blackboard.meta['priorities'] = loss[:, 0].detach()


class MaxLossPriorityComponent(PipelineComponent):
    """
    DQN Standard: Priority is the max error over the entire sequence.
    """
    def __init__(self, loss_key: str = "q_loss"):
        self.loss_key = loss_key

    def execute(self, blackboard: Blackboard) -> None:
        """
        Extracts priorities as the maximum loss over time for each batch element.
        """
        elementwise_losses = blackboard.meta.get("elementwise_losses", {})
        if self.loss_key in elementwise_losses:
            loss = elementwise_losses[self.loss_key]
            # [B, T] -> [B]
            if loss.ndim == 1:
                blackboard.meta['priorities'] = loss.detach()
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

    def execute(self, blackboard: Blackboard) -> None:
        """
        Computes MSE of expected value error at root and stores in blackboard.meta.
        """
        if self.pred_key not in blackboard.predictions or self.target_key not in blackboard.targets:
            return

        # 1. Predictions: Distribution -> Expected Scalar Value [B, T]
        pred_logits = blackboard.predictions[self.pred_key]
        pred_scalars = self.value_representation.to_expected_value(pred_logits)

        # 2. Targets: Raw Scalar Value [B, T]
        target_scalars = blackboard.targets[self.target_key]

        # 3. Compute root TD-error (MSE)
        # Note: We assume T dimension exists. If B is [B, bins], we might need unsqueeze.
        # But Blackboard says all tensors conform to [B, T, ...]
        root_pred = pred_scalars[:, 0]
        root_target = target_scalars[:, 0]

        # Use absolute error or squared error for priorities (rely on (z - q)**2)
        error = (root_target - root_pred) ** 2
        blackboard.meta['priorities'] = error.detach()
