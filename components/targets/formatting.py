"""
High-level composite formatting components for the learner target pipeline.

``TargetFormatterComponent`` is a thin wrapper that applies a mapping of
``{target_key: representation}`` by internally composing one
:class:`~components.targets.formatters.TwoHotProjectionComponent` per key.
This replaces the previous inline call to ``rep.format_target(...)`` so that
the projection logic lives in exactly one place.
"""

from __future__ import annotations

import torch
from typing import Any, Dict

from core import PipelineComponent, Blackboard
from core.path_resolver import resolve_blackboard_path
from modules.representations import BaseRepresentation, DiscreteSupportRepresentation
from components.targets.formatters import TwoHotProjectionComponent



class UniversalInfrastructureComponent(PipelineComponent):
    """Standard Infrastructure Component for single-step learners.

    Ensures masks, importance-sampling weights, and gradient scales are present
    on the blackboard before the loss components execute.  If any of these keys
    are missing they are filled with sensible unit defaults so that downstream
    components do not need to guard for their absence.
    """

    def execute(self, blackboard: Blackboard) -> None:
        # Determine batch size and device from any available tensor
        any_tensor = None
        for section in [blackboard.targets, blackboard.data, blackboard.predictions]:
            if section:
                any_tensor = next((v for v in section.values() if torch.is_tensor(v)), None)
                if any_tensor is not None:
                    break
        
        if any_tensor is None:
            return

        batch_size = any_tensor.shape[0]
        device = any_tensor.device

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
