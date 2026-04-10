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


class TargetFormatterComponent(PipelineComponent):
    """Formats every registered target key using its paired representation.

    For each ``(key, representation)`` pair provided at construction time the
    component converts the raw scalar or distribution stored at
    ``blackboard.targets[key]`` into the representation expected by the loss
    engine.

    For :class:`~learner.losses.representations.DiscreteSupportRepresentation`
    targets (MuZero values, C51, …) the conversion is **delegated** to a
    :class:`~components.targets.formatters.TwoHotProjectionComponent` so the
    projection maths live in exactly one place.

    For all other :class:`~learner.losses.representations.BaseRepresentation`
    subclasses the component falls back to ``representation.format_target``,
    preserving backward-compatibility with policy / to-play heads that use
    ``ClassificationRepresentation`` etc.

    Args:
        target_mapping: A dict mapping blackboard target key → representation.
            Example::

                {
                    "values":   DiscreteSupportRepresentation(-10, 10, 21),
                    "policies": ClassificationRepresentation(num_actions),
                }
    """

    def __init__(self, target_mapping: Dict[str, BaseRepresentation]) -> None:
        self._target_mapping: Dict[str, BaseRepresentation] = target_mapping

        # Pre-build one TwoHotProjectionComponent per discrete-support key.
        self._projectors: Dict[str, TwoHotProjectionComponent] = {}
        for path, rep in target_mapping.items():
            dest_key = path.split(".")[-1]
            if isinstance(rep, DiscreteSupportRepresentation):
                self._projectors[dest_key] = TwoHotProjectionComponent(
                    representation=rep,
                    source_key=dest_key,
                    dest_key=dest_key,
                )

    def execute(self, blackboard: Blackboard) -> None:
        """Convert each registered target key into its loss-ready form."""
        for path, rep in self._target_mapping.items():
            try:
                # 1. Source explicitly from path
                val = resolve_blackboard_path(blackboard, path)
                dest_key = path.split(".")[-1]
                
                # 2. Place in targets for downstream loss components
                blackboard.targets[dest_key] = val

                # 3. Apply Representation-specific formatting
                if dest_key in self._projectors:
                    self._projectors[dest_key].execute(blackboard)
                else:
                    blackboard.targets[dest_key] = rep.format_target(
                        blackboard.targets, target_key=dest_key
                    )
            except KeyError:
                continue


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
