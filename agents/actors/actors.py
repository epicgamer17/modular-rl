import time
import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

from replay_buffers.sequence import Sequence
from replay_buffers.transition import Transition, TransitionBatch
from agents.policies.policy import Policy
from agents.action_selectors.selectors import BaseActionSelector
from utils.wrappers import wrap_recording


class BaseActor(ABC):
    """
    Abstract base class for all actors.
    Handles core game loop and step-level transition collection.
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        agent_network: Any,
        action_selector: BaseActionSelector,
        num_players: Optional[int] = None,
        config: Optional[Any] = None,
        device: Optional[torch.device] = None,
        worker_id: int = 0,
    ):
        """
        Initializes the BaseActor.

        Args:
            env_factory: Factory function to create the environment.
            agent_network: Neural network for value/policy estimation.
            action_selector: Strategy for selecting actions from network output.
            num_players: Number of players.
        """
        self.env_factory = env_factory
        self.agent_network = agent_network
        self.selector = action_selector
        self.config = config
        self.worker_id = worker_id
        self.env = env_factory()
        self.device = device

        # Determine num_players if not provided
        if num_players is not None:
            self.num_players = num_players
        else:
            self.num_players = self._detect_num_players()

        # State for step-level collection
        self._state = None
        self._info = None
        self._done = True  # Start as done to trigger reset
        self._episode_reward = 0.0
        self._episode_length = 0

    @abstractmethod
    def _detect_num_players(self) -> int:
        """Actor-specific player count detection."""
        pass

    @abstractmethod
    def _get_player_id(self) -> Optional[str]:
        """Returns current player ID for multi-player, None for single-player."""
        pass

    @abstractmethod
    def _finalize_episode_info(self, sequence: Sequence) -> None:
        """Add env-specific info to sequence at episode end."""
        pass

    @abstractmethod
    def _get_score(self, sequence: Sequence) -> float:
        """Calculate episode score in env-specific way."""
        pass

    def setup(self):
        """Re-initializes the environment."""
        self.env = self.env_factory()

        # Wrap with RecordVideo if enabled in config and we are the first worker
        if (
            self.config is not None
            and hasattr(self.config, "record_video")
            and self.config.record_video
            and self.worker_id == 0
        ):
            interval = getattr(self.config, "record_video_interval", 1000)
            self.env = wrap_recording(
                self.env,
                video_folder=f"videos/{getattr(self.config, 'model_name', 'agent')}",
                episode_trigger=lambda ep_id: ep_id % interval == 0,
            )

        self._done = True

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Resets the environment and actor state.

        Returns:
            Initial observation and info dictionary.
        """
        self._state, self._info = self._reset_env()
        if self._info is None:
            self._info = {}
        self._done = False
        self._episode_reward = 0.0
        self._episode_length = 0
        return self._state, self._info

    @abstractmethod
    def _reset_env(self) -> Tuple[Any, Dict[str, Any]]:
        """Environment-specific reset logic."""
        pass

    @abstractmethod
    def _step_env(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Environment-specific step logic.

        Returns:
            Tuple of (next_obs, reward, terminated, truncated, info).
        """
        pass

    def step(self) -> Dict[str, Any]:
        """
        Performs one interaction step with the environment.

        Returns:
            A dictionary containing transition details.
        """
        with torch.inference_mode():  # ADD THIS
            if self._done:
                self.reset()

            player_id = self._get_player_id()

            # Convert observation to tensor
            obs_tensor = torch.as_tensor(
                self._state, dtype=torch.float32, device=self.device
            )

            # Determine expected input shape
            expected_shape = self.agent_network.input_shape
            if obs_tensor.dim() == len(expected_shape):
                obs_tensor = obs_tensor.unsqueeze(0)

            action, metadata = self.selector.select_action(
                agent_network=self.agent_network,
                obs=obs_tensor,
                info=self._info,
                player_id=player_id,
                episode_step=self._episode_length,  # Pass loop state
            )
            action_val = action.item() if torch.is_tensor(action) else action

            next_obs, reward, term, trunc, next_info = self._step_env(action_val)

            if next_info is None:
                next_info = {}

            self._done = term or trunc
            self._episode_reward += reward
            self._episode_length += 1

            transition_info = {
                "state": self._state,
                "action": action_val,
                "reward": reward,
                "next_state": next_obs,
                "done": self._done,
                "terminated": bool(term),
                "truncated": bool(trunc),
                "info": self._info,
                "next_info": next_info,
                "player_id": player_id,
                "metadata": metadata,
            }

            self._state = next_obs
            self._info = next_info

            return transition_info

    def play_sequence(self, stats_tracker: Optional[Any] = None) -> Sequence:
        """
        Runs one complete episode and returns a Sequence object.

        Args:
            stats_tracker: Optional statistics tracker for logging.

        Returns:
            A Sequence object containing the episode transcript.
        """
        start_time = time.time()
        sequence = Sequence(self.num_players)

        state, info = self.reset()
        sequence.append(state, info, terminated=False, truncated=False)

        while not self._done:
            player_id = self._get_player_id()
            transition = self.step()

            metadata = transition["metadata"]

            # Extract policies/values from metadata if available
            # This logic depends on what decorators inject
            policy_mode = metadata.get("policy", None)
            # If multi-agent policy info is nested
            if "policies" in metadata and player_id:
                policy_mode = metadata["policies"].get(player_id, policy_mode)

            sequence.append(
                observation=transition["next_state"],
                info=transition["next_info"],
                terminated=transition["terminated"],
                truncated=transition["truncated"],
                action=transition["action"],
                reward=transition["reward"],
                policy=policy_mode,
                value=metadata.get("value"),
                player_id=player_id,
            )

        sequence.duration_seconds = time.time() - start_time
        self._finalize_episode_info(sequence)

        if stats_tracker:
            self._update_stats(sequence, stats_tracker)

        return sequence

    def collect_transitions(
        self, n_transitions: int = 1, stats_tracker: Optional[Any] = None
    ) -> TransitionBatch:
        """
        Collects n transitions and returns them as a batch.

        Args:
            n_transitions: Number of transitions to collect.
            stats_tracker: Optional statistics tracker for logging.

        Returns:
            A TransitionBatch object.
        """
        start_time = time.time()
        transitions: List[Transition] = []
        episodes_completed = 0

        for _ in range(n_transitions):
            transition = self.step()

            transitions.append(
                Transition(
                    observation=transition["state"],
                    action=transition["action"],
                    reward=float(transition["reward"]),
                    next_observation=transition["next_state"],
                    done=transition["done"],
                    terminated=transition["terminated"],
                    truncated=transition["truncated"],
                    info=transition["info"],
                    next_info=transition["next_info"],
                    metadata=transition.get("metadata"),
                )
            )

            if transition["done"]:
                if stats_tracker:
                    stats_tracker.append("score", self._episode_reward)
                    stats_tracker.append("episode_length", self._episode_length)
                    stats_tracker.increment_steps(self._episode_length)
                episodes_completed += 1

        duration = time.time() - start_time
        if stats_tracker and duration > 0:
            stats_tracker.append("actor_fps", len(transitions) / duration)

        return TransitionBatch(
            transitions=transitions,
            episode_stats={
                "episodes_completed": episodes_completed,
                "duration_seconds": duration,
            },
        )

    def _update_stats(self, sequence: Sequence, stats_tracker: Any):
        """Internal helper to log sequence stats."""
        if sequence.duration_seconds > 0:
            stats_tracker.append("actor_fps", len(sequence) / sequence.duration_seconds)

        score = self._get_score(sequence)
        stats_tracker.append("score", score)
        stats_tracker.append("episode_length", len(sequence))
        stats_tracker.increment_steps(len(sequence))

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        """
        Updates the internal parameters of the actor's components.
        Forwards updates to the action selector.
        """
        self.selector.update_parameters(params_dict)


def get_actor_class(env: Any) -> type[BaseActor]:
    """
    Determines the appropriate actor class for a given environment instance.

    Args:
        env: An environment instance.

    Returns:
        The actor class (GymActor or PettingZooActor).
    """
    # Check both the wrapper and the unwrapped environment for PettingZoo indicators
    # PettingZoo AEC environments have 'possible_agents'
    is_pz = hasattr(env, "possible_agents")
    if not is_pz and hasattr(env, "unwrapped"):
        unwrapped = env.unwrapped
        is_pz = hasattr(unwrapped, "possible_agents")

    if is_pz:
        return PettingZooActor

    # Default to GymActor for standard Gymnasium environments
    return GymActor


class GymActor(BaseActor):
    """Actor specialized for Gymnasium single-player environments."""

    def _detect_num_players(self) -> int:
        if self.num_players is not None and self.num_players != 1:
            raise ValueError(f"GymActor requires num_players=1, got {self.num_players}")
        return 1

    def _get_player_id(self) -> None:
        return None

    def _finalize_episode_info(self, sequence: Sequence) -> None:
        pass

    def _get_score(self, sequence: Sequence) -> float:
        return sum(sequence.rewards)

    def _reset_env(self) -> Tuple[Any, Dict[str, Any]]:
        result = self.env.reset()

        # Handle different Gymnasium reset() return formats
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return obs, info or {}

        # Handle older Gym API (obs only) or misidentified PettingZoo (None)
        return result, {}

    def _step_env(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, term, trunc, info = self.env.step(action)
        return obs, float(reward), term, trunc, info


class PettingZooActor(BaseActor):
    """Actor specialized for PettingZoo AEC multi-player environments."""

    def _detect_num_players(self) -> int:
        return len(self.env.possible_agents)

    def _get_player_id(self) -> Optional[str]:
        return self.env.agent_selection

    def _finalize_episode_info(self, sequence: Sequence) -> None:
        sequence.info_history[-1]["final_rewards"] = dict(self.env.rewards)

    def _get_score(self, sequence: Sequence) -> float:
        final_rewards = dict(self.env.rewards)
        agent_id = self.env.possible_agents[0] if self.env.possible_agents else "player_0"
        return final_rewards.get(agent_id, 0.0)

    def _reset_env(self) -> Tuple[Any, Dict[str, Any]]:
        self.env.reset()
        obs, reward, term, trunc, info = self.env.last()
        return obs, info

    def _step_env(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        player_id = self.env.agent_selection
        self.env.step(action)
        obs, reward, term, trunc, info = self.env.last()

        # PettingZoo AEC rewards are incremental and can be extracted per-agent
        player_reward = float(self.env.rewards.get(player_id, 0.0))

        # Include all rewards in info for potential trainer-side processing
        if info is None:
            info = {}
        info["all_player_rewards"] = dict(self.env.rewards)

        return obs, player_reward, term, trunc, info
