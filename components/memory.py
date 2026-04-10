import torch
import numpy as np
from typing import Any, Dict, Optional, TYPE_CHECKING
from core import PipelineComponent
from core import Blackboard

if TYPE_CHECKING:
    from data.storage.circular import ModularReplayBuffer


class BufferStoreComponent(PipelineComponent):
    """Pushes a single-step transition to the replay buffer."""

    def __init__(
        self,
        replay_buffer: 'ModularReplayBuffer',
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

    def _resolve(self, blackboard: Blackboard, path: str) -> Any:
        """Resolves a nested dotted path like 'meta.action_metadata.value' from the blackboard."""
        parts = path.split(".")
        # Start at the blackboard root using the first part as a section name (e.g., 'meta', 'data')
        container = getattr(blackboard, parts[0])
        
        # Traverse remaining parts as dictionary keys
        for key in parts[1:]:
            container = container[key]
        return container

    def execute(self, blackboard: Blackboard) -> None:
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


class SequenceBufferComponent(PipelineComponent):
    """Accumulates steps into a ``Sequence`` and stores on episode end."""

    def __init__(
        self,
        replay_buffer: 'ModularReplayBuffer',
        num_players: int = 1,
    ):
        self.replay_buffer = replay_buffer
        self.num_players = num_players
        self._sequence: Any = None  # lazily imported Sequence

    def _ensure_sequence(self) -> None:
        if self._sequence is None:
            from data.samplers.sequence import Sequence
            self._sequence = Sequence(self.num_players)

    def execute(self, blackboard: Blackboard) -> None:
        self._ensure_sequence()

        obs = blackboard.data.get("obs")
        done = blackboard.data.get("done")
        metadata = blackboard.meta.get("action_metadata", {})
        next_info = blackboard.meta.get("next_info", {})

        # On the very first step, record the initial observation
        if len(self._sequence) == 0:
            initial_obs = obs.squeeze(0).cpu().numpy() if torch.is_tensor(obs) else obs
            info = blackboard.data.get("info", {})
            legal = info.get("legal_moves", [])
            self._sequence.append(initial_obs, terminated=False, truncated=False, legal_moves=legal)

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
        policy = metadata.get("target_policies", metadata.get("policy"))
        if policy is None:
            policy = np.zeros(self.replay_buffer.buffers["policies"].shape[1:], dtype=np.float32)
            
        value = metadata.get("value")
        if value is None:
            value = 0.0
            
        player_id = blackboard.data.get("player_id") # MuZero needs this

        # If terminated/truncated are missing, use done
        if terminated is None and truncated is None:
            is_done = done.item() if torch.is_tensor(done) else done
            terminated = is_done
            truncated = False

        if action is not None:
            self._sequence.append(
                observation=next_obs,
                terminated=terminated,
                truncated=truncated,
                action=action,
                reward=reward,
                policy=policy,
                value=value,
                player_id=player_id,
                legal_moves=next_legal,
            )

        # Flush on episode end
        is_done = done.item() if torch.is_tensor(done) else done
        if is_done:
            self.replay_buffer.store_aggregate(self._sequence)
            # Reset for next episode
            from data.samplers.sequence import Sequence
            self._sequence = Sequence(self.num_players)


class PriorityBufferUpdateComponent(PipelineComponent):
    """Updates priorities in the replay buffer based on learner results."""
    def __init__(self, priority_update_fn: Any):
        self.priority_update_fn = priority_update_fn

    def execute(self, blackboard: Blackboard) -> None:
        priorities = blackboard.meta.get("priorities")
        indices = blackboard.data.get("indices")
        ids = blackboard.data.get("ids")
        
        if priorities is not None and indices is not None:
            # We must move to CPU before sending to the buffer
            self.priority_update_fn(indices, priorities.detach().cpu(), ids=ids)


class PriorityUpdateComponent(PipelineComponent):
    """Computes priorities based on stored elementwise losses."""

    def __init__(self, priority_computer: Any):
        self.priority_computer = priority_computer

    def execute(self, blackboard: Blackboard) -> None:
        elementwise_losses = blackboard.meta.get("elementwise_losses", {})
        if not elementwise_losses:
            return

        priorities = self.priority_computer.compute(
            elementwise_losses=elementwise_losses,
            predictions=blackboard.predictions,
            targets=blackboard.targets,
        )

        blackboard.meta["priorities"] = priorities.detach()


class BetaScheduleComponent(PipelineComponent):
    """Steps the PER beta schedule."""
    def __init__(self, per_beta_schedule: Any, set_beta_fn: Any):
        self.per_beta_schedule = per_beta_schedule
        self.set_beta_fn = set_beta_fn

    def execute(self, blackboard: Blackboard) -> None:
        step = blackboard.meta.get("training_step")
        self.set_beta_fn(self.per_beta_schedule.get_value(step=step))
