from typing import Protocol, Dict
import torch


class PriorityComputer(Protocol):
    """Protocol for calculating replay buffer priorities."""

    def __call__(
        self,
        predictions: dict,
        targets: dict,
        elementwise_losses: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns:
            priority_weights: [B] shaped tensor.
        """
        ...


class SpecificLossPriority:
    """Uses a specific named loss at step 0 for prioritization (MuZero style)."""

    def __init__(self, loss_name: str = "ValueLoss"):
        self.loss_name = loss_name

    def __call__(self, predictions, targets, elementwise_losses) -> torch.Tensor:
        loss = elementwise_losses.get(self.loss_name)
        assert (
            loss is not None
        ), f"PriorityComputer: Loss '{self.loss_name}' not found. Available: {list(elementwise_losses.keys())}"

        # [B, T] -> [B] (slice step 0)
        if loss.ndim == 2:
            return loss[:, 0].detach()
        return loss.detach()


class ErrorPriority:
    """Computes priority based on absolute TD error (DQN/MuZero style)."""

    def __init__(
        self,
        target_key: str = "values",
        prediction_key: str = "values",
        representation=None,
    ):
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.representation = representation

    def __call__(self, predictions, targets, elementwise_losses) -> torch.Tensor:
        pred = predictions[self.prediction_key]
        target = targets[self.target_key]

        # 1. Normalize to root step (k=0)
        p0 = pred[:, 0] if pred.ndim >= 2 else pred
        t0 = target[:, 0] if target.ndim >= 2 else target

        # 2. Convert to scalar expected value if necessary
        if self.representation is not None:
            p0 = self.representation.to_scalar(p0)

        # 3. Handle action selection if pred is [B, Actions]
        if p0.ndim > t0.ndim:
            if "actions" in targets:
                actions = targets["actions"]
                a0 = (actions[:, 0] if actions.ndim >= 2 else actions).long()
                p0 = p0[torch.arange(p0.shape[0], device=p0.device), actions]
            else:
                p0 = p0.squeeze(-1)

        return (p0 - t0).abs().detach()
