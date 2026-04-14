import torch
from core import PipelineComponent, Blackboard
from core.path_resolver import resolve_blackboard_path
from core.contracts import Key, ValueTarget, Reward, Action, Mask, Return, ToPlay, SemanticType, PolicyLogits
from typing import List, Optional, Set, Dict


class SequencePadderComponent(PipelineComponent):
    """Modifier: Pads transition-aligned data to state-aligned length."""

    def __init__(self, unroll_steps: int, keys: List[Key]):
        self.T = unroll_steps + 1
        self._keys = keys
        
        # Deterministic contracts computed at initialization
        self._requires = set(keys)
        self._provides = {
            Key(f"targets.{k.path.split('.')[-1]}", k.semantic_type): "new"
            for k in keys
        }

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        updates = {}
        for k in self._keys:
            key_path = k.path
            try:
                v = resolve_blackboard_path(blackboard, key_path)
                dest_key = key_path.split(".")[-1]
                
                if torch.is_tensor(v) and v.ndim >= 2:
                    current_len = v.shape[1]
                    if current_len == self.T - 1:
                        # Case 1: Unpadded sequence (length K). Pad index 0.
                        padding_shape = list(v.shape)
                        padding_shape[1] = 1
                        padding = torch.zeros(padding_shape, device=v.device, dtype=v.dtype)
                        updates[f"targets.{dest_key}"] = torch.cat([padding, v], dim=1)
                    elif current_len == self.T:
                        # Case 2: Already state-aligned (length K+1). Preserve as is.
                        updates[f"targets.{dest_key}"] = v
                    else:
                        raise AssertionError(
                            f"K+1 Indexing Contract violation on '{key_path}': "
                            f"Expected length {self.T-1} or {self.T}, got {current_len}"
                        )
                else:
                    updates[f"targets.{dest_key}"] = v
            except KeyError:
                continue
        return updates


class SequenceMaskComponent(PipelineComponent):
    """Modifier: Generates or moves [B, T] sequence masks to blackboard.targets.
    
    If masks already exist in blackboard.data (e.g. from NStepUnrollProcessor),
    it moves them to blackboard.targets. Otherwise, it generates them from is_same_game.
    """

    def __init__(self):
        self._requires = {Key("data.is_same_game", Mask)}
        self._provides = {
            Key("targets.value_mask", Mask): "new",
            Key("targets.reward_mask", Mask): "new",
            Key("targets.to_play_mask", Mask): "new",
            Key("targets.policy_mask", Mask): "new",
        }

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        data = blackboard.data
        targets = blackboard.targets
        updates = {}
        
        # 1. Handle Value Mask (States)
        current_value_mask = targets.get("value_mask")
        if current_value_mask is None:
            if "value_mask" in data:
                updates["targets.value_mask"] = data["value_mask"].clone()
            elif "is_same_game" in data:
                updates["targets.value_mask"] = data["is_same_game"].clone()
        
        # 2. Handle Reward Mask (Transitions)
        current_reward_mask = targets.get("reward_mask")
        if current_reward_mask is None:
            if "reward_mask" in data:
                current_reward_mask = data["reward_mask"].clone()
                updates["targets.reward_mask"] = current_reward_mask
            elif "is_same_game" in data:
                current_reward_mask = data["is_same_game"].clone()
                updates["targets.reward_mask"] = current_reward_mask
        
        # Ensure index 0 is masked for rewards ALWAYS
        if current_reward_mask is not None:
            current_reward_mask[:, 0] = False
            assert not current_reward_mask[:, 0].any(), "reward_mask[0] must be False"
            updates["targets.reward_mask"] = current_reward_mask

        # 3. Handle ToPlay Mask
        current_to_play_mask = targets.get("to_play_mask")
        if current_to_play_mask is None:
            if "to_play_mask" in data:
                current_to_play_mask = data["to_play_mask"].clone()
                updates["targets.to_play_mask"] = current_to_play_mask
            elif "is_same_game" in data:
                current_to_play_mask = data["is_same_game"].clone()
                updates["targets.to_play_mask"] = current_to_play_mask
        
        # Ensure index 0 of to_play_mask matches reward_mask (False) per regression tests
        if current_to_play_mask is not None:
            current_to_play_mask[:, 0] = False
            assert not current_to_play_mask[:, 0].any(), "to_play_mask[0] must be False"
            updates["targets.to_play_mask"] = current_to_play_mask

        # 4. Handle Policy Mask
        if "policy_mask" not in targets:
            if "policy_mask" in data:
                updates["targets.policy_mask"] = data["policy_mask"].clone()
            elif "has_valid_obs_mask" in data:
                mask = data["has_valid_obs_mask"].clone()
                if "dones" in data:
                    mask &= ~data["dones"]
                updates["targets.policy_mask"] = mask
        
        return updates


class SequenceInfrastructureComponent(PipelineComponent):
    """Modifier: Generates weight and gradient scale tensors."""

    def __init__(self, unroll_steps: int):
        self.unroll_steps = unroll_steps
        self._requires = {Key("data.actions", Action)}
        self._provides = {
            Key("meta.weights", Mask): "new",
            Key("meta.gradient_scales", Reward): "new"
        }

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        data = blackboard.data
        device = torch.device("cpu")
        for section in [blackboard.targets, blackboard.data, blackboard.predictions]:
            if section:
                any_tensor = next((v for v in section.values() if torch.is_tensor(v)), None)
                if any_tensor is not None:
                    device = any_tensor.device
                    break
        
        B = data["actions"].shape[0]
        updates = {}

        if "weights" not in blackboard.meta:
            updates["meta.weights"] = data.get(
                "weights", torch.ones(B, device=device)
            )

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
        pass

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
