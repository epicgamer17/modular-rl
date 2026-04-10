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
    """Modifier: Generates or moves [B, T] sequence masks to blackboard.targets.
    
    If masks already exist in blackboard.data (e.g. from NStepUnrollProcessor),
    it moves them to blackboard.targets. Otherwise, it generates them from is_same_game.
    """

    def execute(self, blackboard: Blackboard) -> None:
        data = blackboard.data
        
        # 1. Handle Value Mask (States)
        if "value_mask" in data:
            blackboard.targets["value_mask"] = data["value_mask"].clone()
        elif "is_same_game" in data:
            blackboard.targets["value_mask"] = data["is_same_game"].clone()
        
        # 2. Handle Reward Mask (Transitions)
        if "reward_mask" in data:
            blackboard.targets["reward_mask"] = data["reward_mask"].clone()
        elif "is_same_game" in data:
            blackboard.targets["reward_mask"] = data["is_same_game"].clone()
            blackboard.targets["reward_mask"][:, 0] = False
        
        # Ensure index 0 is masked for rewards if it wasn't already
        if "reward_mask" in blackboard.targets:
            blackboard.targets["reward_mask"][:, 0] = False

        # 3. Handle ToPlay Mask
        if "to_play_mask" in data:
            blackboard.targets["to_play_mask"] = data["to_play_mask"].clone()
        elif "is_same_game" in data:
            blackboard.targets["to_play_mask"] = data["is_same_game"].clone()
            blackboard.targets["to_play_mask"][:, 0] = False

        # 4. Handle Policy Mask
        if "policy_mask" in data:
            blackboard.targets["policy_mask"] = data["policy_mask"].clone()
        elif "has_valid_obs_mask" in data:
            blackboard.targets["policy_mask"] = data["has_valid_obs_mask"].clone()
            if "dones" in data:
                blackboard.targets["policy_mask"] &= ~data["dones"]


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
