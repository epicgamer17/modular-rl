import torch
from typing import Dict, List, Optional
from learner.pipeline.base import PipelineComponent
from learner.core import Blackboard


class LossAggregatorComponent(PipelineComponent):
    """
    Flat Loss Pipeline Aggregator.
    Pulls individual scalar losses from blackboard.losses (e.g., 'value_loss', 'policy_loss'),
    sums them according to a mapping, and writes the total to an optimizer-specific key.

    Example:
        mapping = {"default": ["value_loss", "policy_loss", "entropy_loss"]}
        This will create blackboard.losses["default"] = value_loss + policy_loss + entropy_loss.
    """

    def __init__(self, mapping: Optional[Dict[str, List[str]]] = None):
        # Default mapping is to sum everything in blackboard.losses into "default"
        self.mapping = mapping

    def execute(self, blackboard: Blackboard) -> None:
        if not blackboard.losses:
            return

        # Explicit mapping sum
        if self.mapping:
            new_losses = {}
            for opt_key, loss_keys in self.mapping.items():
                total = torch.tensor(
                    0.0, device=next(iter(blackboard.losses.values())).device
                )
                for l_key in loss_keys:
                    if l_key in blackboard.losses:
                        total = total + blackboard.losses[l_key]
                new_losses[opt_key] = total

            # Update blackboard.losses with the aggregated ones
            blackboard.losses.update(new_losses)
        else:
            # Default behavior: Sum everything into "default" NOT ALREADY SUMMED
            # (To avoid double counting if some losses are already optimizer keys)
            device = next(iter(blackboard.losses.values())).device
            total = torch.tensor(0.0, device=device)
            for k, v in blackboard.losses.items():
                total = total + v
            blackboard.losses["default"] = total




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

