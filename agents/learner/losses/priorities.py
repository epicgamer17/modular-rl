import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BasePriorityComputer(ABC):
    """
    Contract for extracting training priorities from loss results.
    Prevents loss modules from knowing about k=0 root priorities or TD-error specifics.
    """

    @abstractmethod
    def compute(
        self,
        elementwise_losses: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns:
            priority [B] tensor
        """
        pass


class NullPriorityComputer(BasePriorityComputer):
    """Returns 1.0 for all batch elements, effectively disabling priority updates."""

    def compute(
        self,
        elementwise_losses: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not elementwise_losses:
            return torch.ones(1)
        B = next(iter(elementwise_losses.values())).shape[0]
        return torch.ones(B, device=next(iter(elementwise_losses.values())).device)


class RootLossPriorityComputer(BasePriorityComputer):
    """
    MuZero Standard: Priority is determined by the loss at the root step (t=0).
    Usually uses 'ValueLoss' as the source.
    """

    def __init__(self, loss_key: str = "ValueLoss"):
        self.loss_key = loss_key

    def compute(
        self,
        elementwise_losses: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.loss_key not in elementwise_losses:
            B = next(iter(elementwise_losses.values())).shape[0]
            return torch.zeros(B, device=next(iter(elementwise_losses.values())).device)

        # elementwise_losses is [B, T]. We want root [B]
        return elementwise_losses[self.loss_key][:, 0].detach()


class MaxLossPriorityComputer(BasePriorityComputer):
    """
    DQN Standard: Priority is the max error over the entire sequence.
    """

    def __init__(self, loss_key: str = "StandardDQNLoss"):
        self.loss_key = loss_key

    def compute(
        self,
        elementwise_losses: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.loss_key not in elementwise_losses:
            B = next(iter(elementwise_losses.values())).shape[0]
            return torch.zeros(B, device=next(iter(elementwise_losses.values())).device)

        # elementwise_losses is [B, T]. We want max over T [B]
        return elementwise_losses[self.loss_key].max(dim=1).values.detach()


class ExpectedValueErrorPriorityComputer(BasePriorityComputer):
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
        pred_key: str = "state_value",
    ):
        self.value_representation = value_representation
        self.target_key = target_key
        self.pred_key = pred_key

    def compute(
        self,
        elementwise_losses: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # 1. Predictions: Distribution -> Expected Scalar Value [B, T]
        # TODO: make this work with distributional AND scalar (ie C51 and standard DQN, or distributional and standard MuZero)
        pred_logits = predictions[self.pred_key]
        pred_scalars = self.value_representation.to_expected_value(pred_logits)

        # 2. Targets: Raw Scalar Value [B, T]
        target_scalars = targets[self.target_key]

        # 3. Compute root TD-error (MSE)
        root_pred = pred_scalars[:, 0]
        root_target = target_scalars[:, 0]

        # Use absolute error or squared error for priorities (rely on (z - q)**2)
        error = (root_target - root_pred) ** 2

        return error.detach()
