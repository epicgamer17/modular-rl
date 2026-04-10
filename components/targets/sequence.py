import torch
from core import PipelineComponent
from core import Blackboard


class SequencePadderComponent(PipelineComponent):
    """Modifier: Pads transition-aligned data to state-aligned length."""

    def __init__(self, unroll_steps: int):
        self.T = unroll_steps + 1

    def execute(self, blackboard: Blackboard) -> None:
        for key, v in blackboard.targets.items():
            if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] == self.T - 1:
                padding_shape = list(v.shape)
                padding_shape[1] = 1
                padding = torch.zeros(padding_shape, device=v.device, dtype=v.dtype)
                blackboard.targets[key] = torch.cat([padding, v], dim=1)


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
        device = (
            next(iter(blackboard.targets.values())).device
            if blackboard.targets
            else torch.device("cpu")
        )
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
