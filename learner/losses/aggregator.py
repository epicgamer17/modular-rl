import torch
from typing import Dict
from learner.pipeline.base import PipelineComponent
from learner.core import Blackboard

class LossAggregatorComponent(PipelineComponent):
    """
    Reads individual loss keys, applies predefined weights, sums them, 
    and writes total_loss to the Blackboard.
    """
    def __init__(self, loss_weights: Dict[str, float], optimizer_key: str = "default"):
        self.loss_weights = loss_weights
        self.optimizer_key = optimizer_key

    def execute(self, blackboard: Blackboard) -> None:
        if not blackboard.losses:
            return

        total_loss = None

        for loss_name, weight in self.loss_weights.items():
            if loss_name in blackboard.losses:
                weighted_loss = weight * blackboard.losses[loss_name]
                if total_loss is None:
                    total_loss = weighted_loss
                else:
                    total_loss = total_loss + weighted_loss

        if total_loss is None:
            return

        # Write the final backward-ready tensor to the blackboard

        # Grouped by optimizer_key in case of disjoint networks (e.g., separate Actor/Critic opts)
        if "total_loss" not in blackboard.losses:
            blackboard.losses["total_loss"] = {}
            
        blackboard.losses["total_loss"][self.optimizer_key] = total_loss


class PriorityUpdateComponent(PipelineComponent):
    """
    New home for priority computation logic.
    Reads elementwise losses (if still available in meta/blackboard) and writes priorities.
    """

    def __init__(self, priority_computer):
        self.priority_computer = priority_computer

    def execute(self, blackboard: Blackboard) -> None:
        """
        Computes priorities based on stored elementwise losses.
        Writes 'priorities' [B] to blackboard.meta for the Replay Buffer Writer.
        """
        elementwise_losses = blackboard.meta.get("elementwise_losses", {})
        if not elementwise_losses:
            return

        # 1. Compute priorities [B]
        # priority_computer.compute returns a tensor of shape [B]
        priorities = self.priority_computer.compute(
            elementwise_losses=elementwise_losses,
            predictions=blackboard.predictions,
            targets=blackboard.targets,
        )

        # 2. Store in blackboard for the Writer component to pick up
        # We must detach and move to cpu to avoid memory leaks in the buffer
        blackboard.meta["priorities"] = priorities.detach()
