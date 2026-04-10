import torch
from typing import Any, Dict
from core import PipelineComponent
from core import Blackboard


class TargetFormatterComponent(PipelineComponent):
    """Formats target keys using their respective representations."""

    def __init__(self, target_mapping: Dict[str, Any]):
        self.target_mapping = target_mapping

    def execute(self, blackboard: Blackboard) -> None:
        for key, rep in self.target_mapping.items():
            if key in blackboard.targets:
                blackboard.targets[key] = rep.format_target(
                    blackboard.targets, target_key=key
                )


class UniversalInfrastructureComponent(PipelineComponent):
    """
    Standard Infrastructure Component for single-step learners.
    Ensures masks, weights, and gradient scales exist.
    """

    def execute(self, blackboard: Blackboard) -> None:
        if not blackboard.targets:
            return

        any_val = next(iter(blackboard.targets.values()))
        batch_size = any_val.shape[0]
        device = any_val.device

        # 1. Generate Universal T=1 Masks if missing
        generic_mask = torch.ones((batch_size, 1), device=device, dtype=torch.bool)
        for mask_key in ["value_mask", "reward_mask", "policy_mask", "q_mask", "masks"]:
            if mask_key not in blackboard.targets:
                blackboard.targets[mask_key] = generic_mask

        # 2. Weights and Gradient Scales
        if "weights" not in blackboard.meta:
            blackboard.meta["weights"] = blackboard.data.get(
                "weights", torch.ones(batch_size, device=device)
            )
        if "gradient_scales" not in blackboard.meta:
            blackboard.meta["gradient_scales"] = torch.ones((1, 1), device=device)
