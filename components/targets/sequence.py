import torch
from core import PipelineComponent, Blackboard
from core.path_resolver import resolve_blackboard_path
from core.contracts import ValueTarget, Reward, Action, Mask, Return, ToPlay # Adjust as needed
from typing import List, Optional


class SequencePadderComponent(PipelineComponent):
    """Modifier: Pads transition-aligned data to state-aligned length."""

    def __init__(self, unroll_steps: int, keys: Optional[List[str]] = None):
        self.T = unroll_steps + 1
        self.keys = keys or ["values", "rewards", "policies", "actions", "to_plays", "reward_mask", "to_play_mask", "action_mask"]

    @property
    def requires(self) -> dict[str, type]:
        # SequencePadder is polymorphic, it just needs the keys to exist.
        # We'll use SemanticType as a base since it can be anything.
        from core.contracts import SemanticType
        return {key: SemanticType for key in self.keys}

    @property
    def provides(self) -> dict[str, type]:
        from core.contracts import SemanticType
        return {f"targets.{key.split('.')[-1]}": SemanticType for key in self.keys}

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
        for key in self.keys:
            try:
                v = resolve_blackboard_path(blackboard, key)
                dest_key = key.split(".")[-1]
                
                if torch.is_tensor(v) and v.ndim >= 2:
                    current_len = v.shape[1]
                    if current_len == self.T - 1:
                        # Case 1: Unpadded sequence (length K). Pad index 0.
                        padding_shape = list(v.shape)
                        padding_shape[1] = 1
                        padding = torch.zeros(padding_shape, device=v.device, dtype=v.dtype)
                        blackboard.targets[dest_key] = torch.cat([padding, v], dim=1)
                    elif current_len == self.T:
                        # Case 2: Already state-aligned (length K+1). Preserve as is.
                        blackboard.targets[dest_key] = v
                    else:
                        raise AssertionError(
                            f"K+1 Indexing Contract violation on '{key}': "
                            f"Expected length {self.T-1} or {self.T}, got {current_len}"
                        )
                else:
                    blackboard.targets[dest_key] = v
            except KeyError:
                continue


class SequenceMaskComponent(PipelineComponent):
    """Modifier: Generates or moves [B, T] sequence masks to blackboard.targets.
    
    If masks already exist in blackboard.data (e.g. from NStepUnrollProcessor),
    it moves them to blackboard.targets. Otherwise, it generates them from is_same_game.
    """

    @property
    def requires(self) -> dict[str, type]:
        return {"data.is_same_game": Mask}

    @property
    def provides(self) -> dict[str, type]:
        return {
            "targets.value_mask": Mask,
            "targets.reward_mask": Mask,
            "targets.to_play_mask": Mask,
            "targets.policy_mask": Mask,
        }

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
        data = blackboard.data
        targets = blackboard.targets
        
        # 1. Handle Value Mask (States)
        if "value_mask" not in targets:
            if "value_mask" in data:
                targets["value_mask"] = data["value_mask"].clone()
            elif "is_same_game" in data:
                targets["value_mask"] = data["is_same_game"].clone()
        
        # 2. Handle Reward Mask (Transitions)
        # Shifted by Padder: blackboard.targets["reward_mask"] should already exist if padded.
        if "reward_mask" not in targets:
            if "reward_mask" in data:
                targets["reward_mask"] = data["reward_mask"].clone()
            elif "is_same_game" in data:
                targets["reward_mask"] = data["is_same_game"].clone()
        
        # Ensure index 0 is masked for rewards ALWAYS
        if "reward_mask" in targets:
            targets["reward_mask"][:, 0] = False
            assert not targets["reward_mask"][:, 0].any(), "reward_mask[0] must be False"

        # 3. Handle ToPlay Mask
        if "to_play_mask" not in targets:
            if "to_play_mask" in data:
                targets["to_play_mask"] = data["to_play_mask"].clone()
            elif "is_same_game" in data:
                targets["to_play_mask"] = data["is_same_game"].clone()
        
        # Ensure index 0 of to_play_mask matches reward_mask (False) per regression tests
        if "to_play_mask" in targets:
            targets["to_play_mask"][:, 0] = False
            assert not targets["to_play_mask"][:, 0].any(), "to_play_mask[0] must be False"

        # 4. Handle Policy Mask
        if "policy_mask" not in targets:
            if "policy_mask" in data:
                targets["policy_mask"] = data["policy_mask"].clone()
            elif "has_valid_obs_mask" in data:
                targets["policy_mask"] = data["has_valid_obs_mask"].clone()
                if "dones" in data:
                    targets["policy_mask"] &= ~data["dones"]


class SequenceInfrastructureComponent(PipelineComponent):
    """Modifier: Generates weight and gradient scale tensors."""

    def __init__(self, unroll_steps: int):
        self.unroll_steps = unroll_steps

    @property
    def requires(self) -> dict[str, type]:
        return {"data.actions": Action}

    @property
    def provides(self) -> dict[str, type]:
        return {"meta.weights": Mask, "meta.gradient_scales": Reward} # Stand-in types

    def validate(self, blackboard: Blackboard) -> None:
        pass

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

    @property
    def requires(self) -> dict[str, type]:
        return {"targets.values": ValueTarget}

    @property
    def provides(self) -> dict[str, type]:
        return {"targets.chance_values_next": ValueTarget}

    def validate(self, blackboard: Blackboard) -> None:
        pass

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
