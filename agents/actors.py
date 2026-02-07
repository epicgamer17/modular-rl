import time
import torch
from typing import Any, Callable, Dict, Optional
from replay_buffers.game import Game
from agents.policy import Policy


from abc import ABC, abstractmethod


class BaseActor(ABC):
    """
    Base class for all actors.
    """

    @abstractmethod
    def setup(self):
        """Initializes or resets the actor's environment."""
        pass

    @abstractmethod
    def play_game(self, stats_tracker: Optional[Any] = None) -> Game:
        """Runs one episode and returns the Game object."""
        pass

    def run_episode(self, stats_tracker: Optional[Any] = None) -> Game:
        """Runs one episode. Alias for play_game to support Executor patterns."""
        return self.play_game(stats_tracker=stats_tracker)


class GenericActor(BaseActor):
    """
    GenericActor handles the game-playing loop, delegating action selection to a policy.
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        policy: Policy,
        num_players: Optional[int] = None,
    ):
        self.env_factory = env_factory
        self.policy = policy
        self.num_players = num_players
        self.env = env_factory()

    def setup(self):
        """Re-initializes the environment."""
        self.env = self.env_factory()

    def configure_video_recording(self, video_folder: str, checkpoint_interval: int):
        """Configures video recording for the environment."""
        try:
            from wrappers import record_video_wrapper

            self.env.render_mode = "rgb_array"
            self.env = record_video_wrapper(
                self.env,
                video_folder,
                checkpoint_interval,
            )
        except Exception as e:
            print(f"Could not record video: {e}")

    def _reset_episode(self) -> tuple[Any, Any, int]:
        """Resets the environment and returns initial state, info, and current player."""
        if self.num_players != 1:
            self.env.reset()
            state, reward, terminated, truncated, info = self.env.last()
            current_player = self.env.agents.index(self.env.agent_selection)
        else:
            state, info = self.env.reset()
            current_player = 0
        return state, info, current_player

    def _step(
        self, action_val: Any, current_player: int
    ) -> tuple[Any, float, bool, int, Any]:
        """Performs a step in the environment."""
        if self.num_players != 1:
            self.env.step(action_val)
            next_state, _, terminated, truncated, next_info = self.env.last()
            reward = self.env.rewards[self.env.agents[current_player]]
            if not (terminated or truncated):
                current_player = self.env.agents.index(self.env.agent_selection)
        else:
            next_state, reward, terminated, truncated, next_info = self.env.step(
                action_val
            )
        return (
            next_state,
            float(reward),
            terminated or truncated,
            current_player,
            next_info,
        )

    def play_game(self, stats_tracker: Optional[Any] = None) -> Game:
        """
        Runs one episode and returns the Game object.
        """
        start_time = time.time()
        state, info, current_player = self._reset_episode()

        if info is None:
            info = {}

        game = Game(self.num_players)
        self.policy.reset(state)
        game.append(state, info)

        done = False
        while not done:
            action = self.policy.compute_action(state, info)
            action_val = action.item() if torch.is_tensor(action) else action

            next_state, reward, done, current_player, next_info = self._step(
                action_val, current_player
            )

            # Policy might provide additional data for the game buffer (e.g. policy, value)
            policy_info = (
                self.policy.get_info() if hasattr(self.policy, "get_info") else {}
            )

            game.append(
                observation=next_state,
                info=next_info,
                action=action_val,
                reward=reward,
                policy=policy_info.get("policy"),
                value=policy_info.get("value"),
            )

            state, info = next_state, next_info

        if self.num_players != 1:
            # Store final rewards for all potentially interested parties (like the agent)
            # We put it in the last info of the game
            game.info_history[-1]["final_rewards"] = self.env.rewards

        # Store duration for FPS calculation (works across process boundaries)
        game.duration_seconds = time.time() - start_time

        if stats_tracker:
            if game.duration_seconds > 0:
                stats_tracker.append("actor_fps", len(game) / game.duration_seconds)

            # Log score (from player 0's perspective)
            if self.num_players == 1:
                score = sum(game.rewards)
            else:
                final_rewards = self.env.rewards
                # Assume the first agent is the one we track stats for
                agent_id = self.env.possible_agents[0]
                score = final_rewards.get(agent_id)

            stats_tracker.append("score", score)
            stats_tracker.append("episode_length", len(game))
            stats_tracker.increment_steps(len(game))

        return game

    def run_episode(self, stats_tracker: Optional[Any] = None) -> Game:
        """Runs one episode. Alias for play_game to support Executor patterns."""
        return self.play_game(stats_tracker=stats_tracker)
