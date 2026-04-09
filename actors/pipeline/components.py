"""Actor pipeline components for the BlackboardEngine ECS.

These components let you compose an actor (environment interaction loop)
using the exact same BlackboardEngine that drives the learner.  The
"batch iterator" for an actor is just an infinite tick generator::

    def infinite_ticks():
        while True:
            yield {}

    actor_engine = BlackboardEngine(actor_pipeline, device=torch.device("cpu"))
    for output in actor_engine.step(infinite_ticks()):
        ...  # output["meta"] contains episode stats when an episode ends

Components
----------
EnvObservationComponent   – writes current obs to blackboard, handles reset
ActorInferenceComponent   – runs obs_inference via a PolicySource
ActionSelectionComponent  – selects action via a BaseActionSelector
EnvStepComponent          – steps the env, writes transition data
BufferStoreComponent      – pushes single-step transitions to replay buffer
SequenceBufferComponent   – accumulates a Sequence, stores on episode end
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch

from learner.pipeline.base import PipelineComponent

if TYPE_CHECKING:
    from learner.core import Blackboard
    from actors.action_selectors.policy_sources import BasePolicySource
    from actors.action_selectors.selectors import BaseActionSelector
    from data.storage.circular import ModularReplayBuffer


# ---------------------------------------------------------------------------
# Shared mutable state between EnvObservation / EnvStep components
# ---------------------------------------------------------------------------

class EnvironmentState:
    """Shared state for environment components in one actor pipeline.

    Both ``EnvObservationComponent`` and ``EnvStepComponent`` hold a
    reference to the same ``EnvironmentState`` so they can coordinate
    without reaching into each other's internals.
    """

    def __init__(
        self,
        env: Any,
        device: torch.device,
        num_actions: int,
        input_shape: tuple[int, ...],
    ):
        self.env = env
        self.device = device
        self.num_actions = num_actions
        self.input_shape = input_shape

        self.obs: Any = None
        self.info: Dict[str, Any] = {}
        self.done: bool = True
        self.episode_reward: float = 0.0
        self.episode_length: int = 0


# ---------------------------------------------------------------------------
# Helper – legal moves sanitisation (mirrors BaseActor._sanitize_boundary_data)
# ---------------------------------------------------------------------------

def _sanitize_info(
    info: Optional[Dict[str, Any]],
    num_actions: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Converts raw env info into pre-vectorised tensors for selectors."""
    if info is None:
        info = {}

    # Legal moves → boolean mask [num_actions]
    mask = torch.zeros(num_actions, dtype=torch.bool, device=device)
    legal = info.get("legal_moves", [])
    if isinstance(legal, (list, np.ndarray, torch.Tensor)) and len(legal) > 0:
        mask[legal] = True
    else:
        mask.fill_(True)
    info["legal_moves"] = mask
    info["legal_moves_mask"] = mask

    return info


# ---------------------------------------------------------------------------
# Environment components
# ---------------------------------------------------------------------------

class EnvObservationComponent(PipelineComponent):
    """Writes the current observation to the blackboard.

    On the first tick (or after a terminal step), resets the environment.
    Always writes ``data["observations"]`` as a ``[1, *obs_shape]`` tensor
    and ``meta["info"]`` with sanitised legal-moves masks.
    """

    def __init__(self, env_state: EnvironmentState):
        self.state = env_state

    def execute(self, blackboard: Blackboard) -> None:
        if self.state.done:
            obs, info = self.state.env.reset()
            if isinstance(obs, tuple):
                obs, info = obs  # handle older gym API
            self.state.obs = obs
            self.state.info = _sanitize_info(
                info, self.state.num_actions, self.state.device
            )
            self.state.done = False
            self.state.episode_reward = 0.0
            self.state.episode_length = 0

        obs_tensor = torch.as_tensor(
            self.state.obs, dtype=torch.float32, device=self.state.device
        )
        if obs_tensor.dim() == len(self.state.input_shape):
            obs_tensor = obs_tensor.unsqueeze(0)

        blackboard.data["observations"] = obs_tensor
        blackboard.meta["info"] = self.state.info


# ---------------------------------------------------------------------------
# Inference & action selection
# ---------------------------------------------------------------------------

class ActorInferenceComponent(PipelineComponent):
    """Runs ``obs_inference`` via a ``PolicySource`` and writes the result.

    Writes ``predictions["inference_result"]`` (an ``InferenceResult``).
    Uses ``torch.inference_mode`` internally for throughput.
    """

    def __init__(self, policy_source: BasePolicySource):
        self.policy_source = policy_source

    def execute(self, blackboard: Blackboard) -> None:
        obs = blackboard.data["observations"]
        info = blackboard.meta.get("info", {})

        with torch.inference_mode():
            result = self.policy_source.get_inference(obs=obs, info=info)

        blackboard.predictions["inference_result"] = result


class ActionSelectionComponent(PipelineComponent):
    """Selects an action using a ``BaseActionSelector``.

    Reads ``predictions["inference_result"]`` and ``meta["info"]``.
    Writes:
    - ``meta["action"]``          – int scalar for ``env.step()``
    - ``meta["action_tensor"]``   – tensor for buffer storage
    - ``meta["action_metadata"]`` – dict (log_prob, value, policy, etc.)
    """

    def __init__(self, action_selector: BaseActionSelector, exploration: bool = True):
        self.action_selector = action_selector
        self.exploration = exploration

    def execute(self, blackboard: Blackboard) -> None:
        result = blackboard.predictions["inference_result"]
        info = blackboard.meta.get("info", {})

        with torch.inference_mode():
            action, metadata = self.action_selector.select_action(
                result=result,
                info=info,
                exploration=self.exploration,
            )

        # Merge extra metadata from the policy source (search stats, etc.)
        if result.extra_metadata:
            for k, v in result.extra_metadata.items():
                if k not in metadata or metadata[k] is None:
                    metadata[k] = v

        # Fallback: if selector didn't produce a policy, use probs from source
        if metadata.get("policy") is None and result.probs is not None:
            metadata["policy"] = result.probs

        # Fallback: value from source (search root value or network value)
        if metadata.get("value") is None and result.value is not None:
            metadata["value"] = result.value

        # Squeeze [1, ...] tensors for single-env actors
        for k, v in metadata.items():
            if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1:
                metadata[k] = v.squeeze(0)

        blackboard.meta["action"] = action.item()
        blackboard.meta["action_tensor"] = action
        blackboard.meta["action_metadata"] = metadata


# ---------------------------------------------------------------------------
# Environment step
# ---------------------------------------------------------------------------

class EnvStepComponent(PipelineComponent):
    """Steps the environment with the selected action.

    Reads ``meta["action"]``.  Writes transition data to ``data`` and
    updates the shared ``EnvironmentState``.  When an episode ends,
    writes ``meta["episode_score"]`` and ``meta["episode_length"]``.
    """

    def __init__(self, env_state: EnvironmentState):
        self.state = env_state

    def execute(self, blackboard: Blackboard) -> None:
        action = blackboard.meta["action"]

        next_obs, reward, terminated, truncated, next_info = self.state.env.step(action)
        done = terminated or truncated

        blackboard.data["rewards"] = torch.tensor(float(reward), device=self.state.device)
        blackboard.data["dones"] = torch.tensor(done, device=self.state.device)
        blackboard.data["terminated"] = torch.tensor(terminated, device=self.state.device)
        blackboard.data["truncated"] = torch.tensor(truncated, device=self.state.device)
        blackboard.data["next_observations"] = torch.as_tensor(
            next_obs, dtype=torch.float32, device=self.state.device
        )
        blackboard.meta["next_info"] = next_info

        self.state.episode_reward += float(reward)
        self.state.episode_length += 1

        if done:
            blackboard.meta["episode_score"] = self.state.episode_reward
            blackboard.meta["episode_length"] = self.state.episode_length

        self.state.obs = next_obs
        self.state.info = _sanitize_info(
            next_info, self.state.num_actions, self.state.device
        )
        self.state.done = done


# ---------------------------------------------------------------------------
# Buffer storage
# ---------------------------------------------------------------------------

class BufferStoreComponent(PipelineComponent):
    """Pushes a single-step transition to the replay buffer.

    Reads standard keys from ``batch`` and ``meta`` and calls
    ``replay_buffer.store(**transition)``.  The ``field_map`` dict
    controls which blackboard keys map to which buffer field names.
    """

    def __init__(
        self,
        replay_buffer: ModularReplayBuffer,
        field_map: Optional[Dict[str, str]] = None,
    ):
        self.replay_buffer = replay_buffer
        self.field_map = field_map or {
            "observations": "data.observations",
            "actions": "meta.action",
            "rewards": "data.rewards",
            "dones": "data.dones",
            "next_observations": "data.next_observations",
        }

    def _resolve(self, blackboard: Blackboard, path: str) -> Any:
        """Resolves a dotted path like 'meta.action' from the blackboard."""
        section, key = path.split(".", 1)
        container = getattr(blackboard, section)
        return container[key]

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
    """Accumulates steps into a ``Sequence`` and stores on episode end.

    For algorithms that require full-episode storage (MuZero, etc.),
    this component collects transitions tick-by-tick and calls
    ``replay_buffer.store_aggregate(sequence)`` when the episode ends.
    """

    def __init__(
        self,
        replay_buffer: ModularReplayBuffer,
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

        obs = blackboard.data.get("observations")
        done = blackboard.data.get("dones")
        metadata = blackboard.meta.get("action_metadata", {})
        next_info = blackboard.meta.get("next_info", {})

        # On the very first step, record the initial observation
        if len(self._sequence) == 0:
            initial_obs = obs.squeeze(0).cpu().numpy() if torch.is_tensor(obs) else obs
            info = blackboard.meta.get("info", {})
            legal = info.get("legal_moves", [])
            self._sequence.append(initial_obs, terminated=False, truncated=False, legal_moves=legal)

        # Record the transition
        next_obs = blackboard.data.get("next_observations")
        if torch.is_tensor(next_obs):
            next_obs = next_obs.cpu().numpy()

        action = blackboard.meta.get("action")
        reward = blackboard.data.get("rewards")
        if torch.is_tensor(reward):
            reward = reward.item()

        terminated = blackboard.data.get("terminated")
        if torch.is_tensor(terminated):
            terminated = terminated.item()
        truncated = blackboard.data.get("truncated")
        if torch.is_tensor(truncated):
            truncated = truncated.item()

        next_legal = next_info.get("legal_moves", [])
        policy = metadata.get("target_policies", metadata.get("policy"))
        value = metadata.get("value")

        self._sequence.append(
            observation=next_obs,
            terminated=terminated,
            truncated=truncated,
            action=action,
            reward=reward,
            policy=policy,
            value=value,
            legal_moves=next_legal,
        )

        # Flush on episode end
        is_done = done.item() if torch.is_tensor(done) else done
        if is_done:
            self.replay_buffer.store_aggregate(self._sequence)
            # Reset for next episode
            from data.samplers.sequence import Sequence
            self._sequence = Sequence(self.num_players)


# ---------------------------------------------------------------------------
# Utility: infinite tick generator
# ---------------------------------------------------------------------------

def infinite_ticks():
    """Batch iterator for actors — yields empty dicts forever."""
    while True:
        yield {}
