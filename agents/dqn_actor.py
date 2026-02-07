"""
DQNActor for collecting transitions (not games) for DQN-style algorithms.
"""

import time
import torch
from typing import Any, Callable, List, Optional

from agents.actors import BaseActor
from agents.policies.policy import Policy
from replay_buffers.transition import Transition, TransitionBatch


class DQNActor(BaseActor):
    """
    DQNActor collects individual transitions for DQN-style algorithms.

    Unlike GenericActor which returns complete Game objects, DQNActor returns
    TransitionBatch objects containing individual (s, a, r, s', done) tuples.
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        policy: Policy,
        num_players: Optional[int] = None,
    ):
        """
        Initializes the DQNActor.

        Args:
            env_factory: Factory function to create the environment.
            policy: Policy object for action selection.
            num_players: Number of players (1 for single-player).
        """
        self.env_factory = env_factory
        self.policy = policy
        self.num_players = num_players if num_players is not None else 1
        self.env = env_factory()

        # Persistent state for mid-episode collection
        self._state = None
        self._info = None
        self._episode_reward = 0.0
        self._episode_length = 0
        self._needs_reset = True

    def setup(self):
        """Re-initializes the environment."""
        self.env = self.env_factory()
        self._needs_reset = True

    def _reset_episode(self) -> tuple[Any, Any]:
        """Resets the environment and returns initial state and info."""
        if self.num_players != 1:
            self.env.reset()
            state, _, _, _, info = self.env.last()
        else:
            state, info = self.env.reset()

        self._state = state
        self._info = info if info is not None else {}
        self._episode_reward = 0.0
        self._episode_length = 0
        self._needs_reset = False
        return state, self._info

    def collect_transitions(
        self, n_transitions: int = 1, stats_tracker: Optional[Any] = None
    ) -> TransitionBatch:
        """
        Collects n transitions and returns them as a batch.

        Args:
            n_transitions: Number of transitions to collect.
            stats_tracker: Optional stats tracker for logging.

        Returns:
            TransitionBatch containing the collected transitions.
        """
        start_time = time.time()
        transitions: List[Transition] = []
        episodes_completed = 0

        for _ in range(n_transitions):
            if self._needs_reset:
                self._reset_episode()
                self.policy.reset(self._state)

            # Select action
            action = self.policy.compute_action(self._state, self._info)
            # print("DEBUG: Action", action)
            action_val = action.item() if torch.is_tensor(action) else action

            # Step environment
            if self.num_players != 1:
                self.env.step(action_val)
                next_state, _, terminated, truncated, next_info = self.env.last()
                current_player = (
                    self.env.agents.index(self.env.agent_selection)
                    if not (terminated or truncated)
                    else 0
                )
                reward = self.env.rewards.get(self.env.agents[0], 0.0)
            else:
                next_state, reward, terminated, truncated, next_info = self.env.step(
                    action_val
                )

            done = terminated or truncated
            if next_info is None:
                next_info = {}

            # Create transition
            transition = Transition(
                observation=self._state,
                action=action_val,
                reward=float(reward),
                next_observation=next_state,
                done=done,
                info=self._info,
                next_info=next_info,
            )
            transitions.append(transition)

            # Update episode stats
            self._episode_reward += reward
            self._episode_length += 1

            if done:
                # Log episode stats
                if stats_tracker:
                    stats_tracker.append("score", self._episode_reward)
                    stats_tracker.append("episode_length", self._episode_length)
                    stats_tracker.increment_steps(self._episode_length)

                episodes_completed += 1
                self._needs_reset = True
            else:
                self._state = next_state
                self._info = next_info

        # Calculate FPS
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

    def play_game(self, stats_tracker: Optional[Any] = None):
        """
        Runs one complete episode and returns transitions.

        For compatibility with BaseActor interface, but returns TransitionBatch
        instead of Game.
        """
        transitions: List[Transition] = []
        self._reset_episode()
        self.policy.reset(self._state)

        start_time = time.time()

        while not self._needs_reset:
            batch = self.collect_transitions(n_transitions=1, stats_tracker=None)
            transitions.extend(batch.transitions)

        duration = time.time() - start_time

        if stats_tracker:
            stats_tracker.append("score", self._episode_reward)
            stats_tracker.append("episode_length", self._episode_length)
            stats_tracker.increment_steps(self._episode_length)
            if duration > 0:
                stats_tracker.append("actor_fps", len(transitions) / duration)

        return TransitionBatch(
            transitions=transitions,
            episode_stats={
                "episode_reward": self._episode_reward,
                "episode_length": self._episode_length,
                "duration_seconds": duration,
            },
        )

    def run_episode(self, stats_tracker: Optional[Any] = None) -> TransitionBatch:
        """Runs one episode. Alias for play_game."""
        return self.play_game(stats_tracker=stats_tracker)
