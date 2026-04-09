import torch
import numpy as np
from typing import Any, Dict, Optional, TYPE_CHECKING
from core import PipelineComponent
from core import Blackboard

if TYPE_CHECKING:
    from actors.action_selectors.selectors import BaseActionSelector


# TODO: split a little better into stuff like action selection vs env interfaces (gym, pettingzoo puffer etc)
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


class ActionSelectionComponent(PipelineComponent):
    """Selects an action using a ``BaseActionSelector``.

    Reads ``predictions["inference_result"]`` and ``meta["info"]``.
    Writes:
    - ``meta["action"]``          – int scalar for ``env.step()``
    - ``meta["action_tensor"]``   – tensor for buffer storage
    - ``meta["action_metadata"]`` – dict (log_prob, value, policy, etc.)
    """

    def __init__(self, action_selector: "BaseActionSelector", exploration: bool = True):
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

        blackboard.data["rewards"] = torch.tensor(
            float(reward), device=self.state.device
        )
        blackboard.data["dones"] = torch.tensor(done, device=self.state.device)
        blackboard.data["terminated"] = torch.tensor(
            terminated, device=self.state.device
        )
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


class EpsilonScheduleComponent(PipelineComponent):
    """Steps the epsilon-greedy exploration schedule."""

    def __init__(self, epsilon_schedule: Any):
        self.epsilon_schedule = epsilon_schedule

    def execute(self, blackboard: Blackboard) -> None:
        self.epsilon_schedule.step()
