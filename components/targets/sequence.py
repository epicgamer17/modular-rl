import torch
from typing import List, Optional
from core import PipelineComponent
from core import Blackboard
from core.path_resolver import resolve_blackboard_path


class SequencePadderComponent(PipelineComponent):
    """Modifier: Pads transition-aligned data to state-aligned length."""

    def __init__(self, unroll_steps: int, keys: Optional[List[str]] = None):
        self.T = unroll_steps + 1
        self.keys = keys or ["values", "rewards", "policies", "actions", "to_plays"]

    def execute(self, blackboard: Blackboard) -> None:
        for key in self.keys:
            try:
                v = resolve_blackboard_path(blackboard, key)
                dest_key = key.split(".")[-1]
                
                if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] == self.T - 1:
                    padding_shape = list(v.shape)
                    padding_shape[1] = 1
                    padding = torch.zeros(padding_shape, device=v.device, dtype=v.dtype)
                    blackboard.targets[dest_key] = torch.cat([padding, v], dim=1)
                else:
                    blackboard.targets[dest_key] = v
            except KeyError:
                continue


class SequenceMaskComponent(PipelineComponent):
    """Modifier: Generates Universal [B, T] sequence masks."""

    def execute(self, blackboard: Blackboard) -> None:
        data = blackboard.data
        blackboard.targets["value_mask"] = data["is_same_game"].clone()
        blackboard.targets["masks"] = data["is_same_game"].clone()
        blackboard.targets["policy_mask"] = data["has_valid_obs_mask"].clone()
        blackboard.targets["policy_mask"] &= ~data["dones"]

        blackboard.targets["reward_mask"] = data["is_same_game"].clone()
        blackboard.targets["reward_mask"][:, 0] = False

        blackboard.targets["to_play_mask"] = data["is_same_game"].clone()
        blackboard.targets["to_play_mask"][:, 0] = False


class SequenceInfrastructureComponent(PipelineComponent):
    """Modifier: Generates weight and gradient scale tensors."""

    def __init__(self, unroll_steps: int):
        self.unroll_steps = unroll_steps

    def execute(self, blackboard: Blackboard) -> None:
        data = blackboard.data
        device = torch.device("cpu")
        for section in [blackboard.targets, blackboard.data, blackboard.predictions]:
            if section:
                any_tensor = next((v for v in section.values() if torch.is_tensor(v)), None)
                if any_tensor is not None:
                    device = any_tensor.device
                    break
        
        B = data["actions"].shape[0]

        if "weights" not in blackboard.meta:
            blackboard.meta["weights"] = data.get(
                "weights", torch.ones(B, device=device)
            )

        if "gradient_scales" not in blackboard.meta:
            scales = (
                [1.0] + [1.0 / self.unroll_steps] * self.unroll_steps
                if self.unroll_steps > 0
                else [1.0]
            )
            blackboard.meta["gradient_scales"] = torch.tensor(
                scales, device=device
            ).reshape(1, -1)


class ChanceTargetComponent(PipelineComponent):
    """Generator: Calculates chance outcomes for Stochastic MuZero."""

    def execute(self, blackboard: Blackboard) -> None:
        # Stochastic MuZero shifts the value target by 1 step for chance nodes
        if (
            "values" in blackboard.targets
            and "chance_values_next" not in blackboard.targets
        ):
            v = blackboard.targets["values"]
            v_next = torch.zeros_like(v)
            v_next[:, :-1] = v[:, 1:]  # Shift left
            blackboard.targets["chance_values_next"] = v_next
