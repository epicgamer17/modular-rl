import torch
import numpy as np
from core import PipelineComponent, Blackboard
from core.contracts import Key, SemanticType, Observation, Action, Reward, Done, Mask
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from data.storage.circular import ModularReplayBuffer


class BufferStoreComponent(PipelineComponent):
    """Pushes a single-step transition to the replay buffer."""

    def __init__(
        self,
        replay_buffer: "ModularReplayBuffer",
        field_map: Optional[Dict[str, str]] = None,
    ):
        self.replay_buffer = replay_buffer
        self.field_map = field_map or {
            "observations": "data.obs",
            "actions": "meta.action",
            "rewards": "data.reward",
            "dones": "data.done",
            "next_observations": "data.next_obs",
        }
        # Deterministic contracts computed at initialization
        self._requires = {Key(bb_path, SemanticType) for bb_path in self.field_map.values()}
        self._provides = {}

    def _resolve(self, blackboard: Blackboard, path: str) -> Any:
        """Resolves a nested dotted path like 'meta.action_metadata.value' from the blackboard."""
        parts = path.split(".")
        # Start at the blackboard root using the first part as a section name (e.g., 'meta', 'data')
        container = getattr(blackboard, parts[0])

        # Traverse remaining parts as dictionary keys
        for key in parts[1:]:
            container = container[key]
        return container

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures all mapped blackboard paths are resolvable."""
        for buffer_key, bb_path in self.field_map.items():
            try:
                self._resolve(blackboard, bb_path)
            except (KeyError, AttributeError):
                assert False, (
                    f"BufferStoreComponent: path '{bb_path}' (for buffer field '{buffer_key}') "
                    f"not resolvable on the blackboard"
                )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        transition = {}
        for buffer_key, bb_path in self.field_map.items():
            val = self._resolve(blackboard, bb_path)
            # Detach tensors to avoid holding onto the compute graph
            if torch.is_tensor(val):
                val = val.detach().cpu()
            transition[buffer_key] = val

        # Merge action metadata if the buffer expects extra fields (value, log_prob, etc.)
        metadata = blackboard.meta.get("action_metadata", {})
        info = blackboard.meta.get("info", {})
        transition["info"] = info

        for k, v in metadata.items():
            if k not in transition:
                if torch.is_tensor(v):
                    transition[k] = v.detach().cpu()
                else:
                    transition[k] = v

        self.replay_buffer.store(**transition)
        return {}


class SequenceBufferComponent(PipelineComponent):
    """Accumulates steps into a ``Sequence`` and stores on episode end."""

    def __init__(
        self,
        replay_buffer: "ModularReplayBuffer",
        num_players: int = 1,
        target_policy_key: Optional[str] = None,
        target_value_key: Optional[str] = None,
    ):
        self.replay_buffer = replay_buffer
        self.num_players = num_players
        self.target_policy_key = target_policy_key
        self.target_value_key = target_value_key
        self._sequence: Any = None  # lazily imported Sequence

        # Deterministic contracts computed at initialization
        from core.contracts import PolicyLogits, ValueEstimate
        self._requires = {
            Key("data.obs", Observation),
            Key("data.done", Done),
            Key("data.reward", Reward),
            Key("meta.action", Action),
        }
        if self.target_policy_key:
            self._requires.add(Key(f"predictions.{self.target_policy_key}", PolicyLogits))
        if self.target_value_key:
            self._requires.add(Key(f"predictions.{self.target_value_key}", ValueEstimate))
        self._provides = {}

    def _ensure_sequence(self) -> None:
        if self._sequence is None:
            from data.samplers.sequence import Sequence

            self._sequence = Sequence(self.num_players)

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures obs and done signals exist."""
        assert blackboard.data.get("obs") is not None, (
            "SequenceBufferComponent: 'obs' missing from blackboard.data"
        )
        assert blackboard.data.get("done") is not None, (
            "SequenceBufferComponent: 'done' missing from blackboard.data"
        )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        self._ensure_sequence()

        obs = blackboard.data.get("obs")
        done = blackboard.data.get("done")
        metadata = blackboard.meta.get("action_metadata", {})
        next_info = blackboard.meta.get("next_info", {})

        # On the very first step, record the initial observation
        if len(self._sequence) == 0:
            initial_obs = obs.squeeze(0).cpu().numpy() if torch.is_tensor(obs) else obs
            info = blackboard.data.get("info", {})
            player_id = blackboard.data.get(
                "player_id", 0 if self.num_players == 1 else None
            )
            legal = info.get("legal_moves", [])
            self._sequence.append(
                initial_obs,
                terminated=False,
                truncated=False,
                player_id=player_id,
                legal_moves=legal,
            )

        # Record the transition
        next_obs = blackboard.data.get("next_obs")
        if next_obs is None and obs is not None:
            # Match shape of current obs if next_obs is missing (terminal state)
            next_obs = torch.zeros_like(obs)

        if torch.is_tensor(next_obs):
            next_obs = next_obs.squeeze(0).cpu().numpy()

        action = blackboard.meta.get("action")
        reward = blackboard.data.get("reward")
        if torch.is_tensor(reward):
            reward = reward.item()

        terminated = blackboard.data.get("terminated", False)
        if torch.is_tensor(terminated):
            terminated = terminated.item()
        truncated = blackboard.data.get("truncated", False)
        if torch.is_tensor(truncated):
            truncated = truncated.item()

        next_legal = next_info.get("legal_moves", [])

        # --- TARGET LOGIC ---
        # Prioritize search/custom targets from predictions if keys are provided
        policy = None
        if self.target_policy_key:
            policy = blackboard.predictions.get(self.target_policy_key)

        if policy is not None and torch.is_tensor(policy):
            policy = policy.detach().cpu().numpy()

        if policy is None:
            policy = np.zeros(
                self.replay_buffer.buffers["policies"].shape[1:], dtype=np.float32
            )

        value = None
        if self.target_value_key:
            value = blackboard.predictions.get(self.target_value_key)

        if value is not None and torch.is_tensor(value):
            value = value.detach().cpu().item()

        if value is None:
            # TODO: remove this? remove all these if checks too?
            value = 0.0

        player_id = blackboard.data.get(
            "player_id", 0 if self.num_players == 1 else None
        )  # MuZero needs this

        # If terminated/truncated are missing, use done
        if terminated is None and truncated is None:
            is_done = done.item() if torch.is_tensor(done) else done
            terminated = is_done
            truncated = False

        next_player_id = blackboard.data.get(
            "next_player_id", 0 if self.num_players == 1 else None
        )

        if action is not None:
            self._sequence.append(
                observation=next_obs,
                terminated=terminated,
                truncated=truncated,
                action=action,
                reward=reward,
                policy=policy,
                value=value,
                player_id=next_player_id,
                legal_moves=next_legal,
            )

        # Flush on episode end
        is_done = done.item() if torch.is_tensor(done) else done
        if is_done:
            self.replay_buffer.store_aggregate(self._sequence)
            # Reset for next episode
            from data.samplers.sequence import Sequence

            self._sequence = Sequence(self.num_players)
        
        return {}
