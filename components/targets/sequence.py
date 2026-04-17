import torch
from core import PipelineComponent, Blackboard
from core.path_resolver import resolve_blackboard_path
from core.contracts import (
    Key,
    ValueTarget,
    Reward,
    Action,
    Mask,
    Return,
    ToPlay,
    SemanticType,
    Policy,
    Weight,
    GradientScale,
)
from typing import Any, Dict, List, Optional, Set


class UnrollGradientScaler(PipelineComponent):
    """Modifier: Generates gradient scale tensors for MuZero-style unrolls."""

    required = True

    def __init__(self, unroll_steps: int):
        self.unroll_steps = unroll_steps
        self._requires = {Key("data.actions", Action)}
        self._provides = {
            Key("meta.gradient_scales", GradientScale): "new",
        }

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures actions exist in data."""
        assert (
            "actions" in blackboard.data
        ), "UnrollGradientScaler: 'actions' missing from blackboard.data"

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        device = blackboard.data["actions"].device
        
        updates = {}
        if "gradient_scales" not in blackboard.meta:
            scales = (
                [1.0] + [1.0 / self.unroll_steps] * self.unroll_steps
                if self.unroll_steps > 0
                else [1.0]
            )
            updates["meta.gradient_scales"] = torch.tensor(
                scales, device=device
            ).reshape(1, -1)

        return updates


class ChanceTargetComponent(PipelineComponent):
    """Generator: Calculates chance outcomes for Stochastic MuZero."""

    def __init__(self):
        self._requires = {Key("targets.values", ValueTarget)}
        self._provides = {Key("targets.chance_values_next", ValueTarget): "new"}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures value targets exist and have a time dimension."""
        from core.validation import assert_is_tensor

        if "values" in blackboard.targets:
            v = blackboard.targets["values"]
            assert_is_tensor(v, msg="in ChanceTargetComponent (targets.values)")
            assert (
                v.ndim >= 2
            ), f"ChanceTargetComponent: values must have [B, T], got {v.shape}"

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        # Stochastic MuZero shifts the value target by 1 step for chance nodes
        if (
            "values" in blackboard.targets
            and "chance_values_next" not in blackboard.targets
        ):
            v = blackboard.targets["values"]
            v_next = torch.zeros_like(v)
            v_next[:, :-1] = v[:, 1:]  # Shift left
            return {"targets.chance_values_next": v_next}
        return {}
