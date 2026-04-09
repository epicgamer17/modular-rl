import torch
import numpy as np
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING
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


# Action selection components have been moved to components/actor_logic.py


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


class SimpleEnvObservationComponent(PipelineComponent):
    """Phase 1: Injects the current environment state into the Blackboard.

    This is a simplified version that manages its own observation state.
    """

    def __init__(self, env: Any):
        self.env = env
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs, _ = obs
        self.current_obs = obs

    def execute(self, blackboard: Blackboard) -> None:
        # Write observation to data dict (ensuring [B, T, ...] format if vectorized)
        blackboard.data["observations"] = torch.as_tensor(
            self.current_obs, dtype=torch.float32
        ).unsqueeze(0)


class SimpleEnvStepComponent(PipelineComponent):
    """Phase 2: Reads the action, steps the environment, and records the transition.

    This component relies on the EnvObservationComponent to update its state for the next tick.
    """

    def __init__(
        self,
        env: Any,
        obs_component: SimpleEnvObservationComponent,
        stop_on_done: bool = False,
    ):
        self.env = env
        self.obs_component = obs_component
        self.stop_on_done = stop_on_done
        self.episode_reward = 0.0
        self.episode_length = 0

    def execute(self, blackboard: Blackboard) -> None:
        if (
            "actions" not in blackboard.predictions
            and "action" not in blackboard.meta
        ):
            raise ValueError(
                "SimpleEnvStepComponent requires 'actions' in blackboard.predictions or 'action' in blackboard.meta"
            )

        if "actions" in blackboard.predictions:
            action = blackboard.predictions["actions"].item()
        else:
            action = blackboard.meta["action"]

        step_result = self.env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_result

        # Record the full transition for the Replay Buffer
        blackboard.data["actions"] = torch.tensor([action])
        blackboard.data["rewards"] = torch.tensor([float(reward)])
        blackboard.data["dones"] = torch.tensor([done])
        blackboard.data["next_observations"] = torch.as_tensor(
            next_obs, dtype=torch.float32
        ).unsqueeze(0)

        blackboard.meta["done"] = done
        blackboard.meta["terminated"] = terminated
        blackboard.meta["info"] = info

        self.episode_reward += float(reward)
        self.episode_length += 1

        # Update the Observation component's state for the next tick
        if done:
            blackboard.meta["episode_score"] = self.episode_reward
            blackboard.meta["episode_length"] = self.episode_length
            if self.stop_on_done:
                blackboard.meta["stop_execution"] = True

            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs, _ = obs
            self.obs_component.current_obs = obs
            self.episode_reward = 0.0
            self.episode_length = 0
        else:
            self.obs_component.current_obs = next_obs


class PettingZooAECComponent(PipelineComponent):
    """
    Component for PettingZoo AEC environments.
    Merges observation and stepping into a single component for turn-based games.
    """

    def __init__(self, env: Any, input_shape: Tuple[int, ...], stop_on_done: bool = True, device: torch.device = torch.device("cpu")):
        self.env = env
        self.input_shape = input_shape
        self.stop_on_done = stop_on_done
        self.device = device
        self.env.reset()
        self._last_reward = 0.0
        self._last_done = False
        self._last_terminated = False
        self._last_truncated = False
        self.episode_rewards: Dict[str, float] = {a: 0.0 for a in self.env.possible_agents}
        self.episode_step = 0

    def execute(self, blackboard: Blackboard) -> None:
        # 1. If an action was chosen in the previous component, step the environment
        if "action" in blackboard.meta:
            action = blackboard.meta["action"]
            self.env.step(action)

        # 2. Observe the state for the current active agent
        agent = self.env.agent_selection
        obs, reward, termination, truncation, info = self.env.last()
        done = termination or truncation

        # 3. Write observation to blackboard
        # For single-agent inference components
        if obs is None and done:
            obs = np.zeros(self.input_shape, dtype=np.float32)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        blackboard.data["observations"] = obs_tensor.unsqueeze(0)
        blackboard.data["active_agent"] = agent
        
        # Determine player index for MuZero
        try:
            player_idx = self.env.possible_agents.index(agent)
        except (ValueError, AttributeError):
            player_idx = 0
        
        blackboard.meta["to_play"] = player_idx
        blackboard.data["player_id"] = player_idx

        # 4. Write transition data (for the previous agent's action)
        # Note: In AEC, 'reward' is the reward the current agent received.
        # MuZero typically expects rewards for all agents or just the active one.
        blackboard.data["rewards"] = torch.tensor(float(reward), device=self.device)
        blackboard.data["dones"] = torch.tensor(done, device=self.device)
        blackboard.data["terminated"] = torch.tensor(termination, device=self.device)
        blackboard.data["truncated"] = torch.tensor(truncation, device=self.device)

        if done and self.stop_on_done:
            blackboard.meta["stop_execution"] = True
            # Reset happens on the NEXT tick's first call to execute() or we can do it here
        
        # Sanitize info for action selection
        num_actions = self.env.action_space(agent).n
        blackboard.meta["info"] = _sanitize_info(info, num_actions, self.device)
        blackboard.meta["info"]["episode_step"] = self.episode_step
        blackboard.meta["done"] = done
        blackboard.meta["terminated"] = termination
        
        # Telemetry
        self.episode_rewards[agent] += float(reward)
        self.episode_step += 1

        if done:
            blackboard.meta[f"episode_score_{agent}"] = self.episode_rewards[agent]
            # Reset for next episode
            self.episode_rewards[agent] = 0.0
            self.episode_step = 0
            
            # Reset environment if ALL agents are done (standard for PZ parallel/sequence wrappers)
            # Actually for AEC, we should check if all agents are done. 
            # But PZ envs often stop yielding agents when the game is over.
            if not self.env.agents:
                self.env.reset()
