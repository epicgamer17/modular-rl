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
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Returns:
            priority [B] tensor
        """
        pass

class NullPriorityComputer(BasePriorityComputer):
    """Returns 1.0 for all batch elements, effectively disabling priority updates."""
    def compute(self, elementwise_losses, predictions, targets, context):
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

    def compute(self, elementwise_losses, predictions, targets, context):
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

    def compute(self, elementwise_losses, predictions, targets, context):
        if self.loss_key not in elementwise_losses:
             B = next(iter(elementwise_losses.values())).shape[0]
             return torch.zeros(B, device=next(iter(elementwise_losses.values())).device)
             
        # elementwise_losses is [B, T]. We want max over T [B]
        return elementwise_losses[self.loss_key].max(dim=1).values.detach()
