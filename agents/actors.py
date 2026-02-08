import time
import torch
from typing import Any, Callable, Dict, Optional, Union
from replay_buffers.game import Game
from agents.policies.policy import Policy


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
        self, action_val: Any, current_player: int, player_id: str = None
    ) -> tuple[Any, Union[float, Dict[str, float]], bool, int, Any]:
        """
        Performs a step in the environment.

        For multi-player games, returns all_rewards dict keyed by player_id.
        For single-player, returns a single float reward.
        """
        if self.num_players != 1:
            self.env.step(action_val)
            next_state, _, terminated, truncated, next_info = self.env.last()
            # Return ALL player rewards so we can assign correct reward per player
            all_rewards = dict(self.env.rewards)
            if not (terminated or truncated):
                current_player = self.env.agents.index(self.env.agent_selection)
            return (
                next_state,
                all_rewards,  # Dict of all player rewards
                terminated or truncated,
                current_player,
                next_info,
            )
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
            # Get player_id for multi-player environments
            if self.num_players != 1 and hasattr(self.env, "agent_selection"):
                player_id = self.env.agent_selection
            else:
                player_id = None

            action = self.policy.compute_action(state, info, player_id=player_id)
            action_val = action.item() if torch.is_tensor(action) else action

            next_state, reward_data, done, current_player, next_info = self._step(
                action_val, current_player, player_id
            )

            # Policy might provide additional data for the game buffer (e.g. policy, value)
            policy_info = (
                self.policy.get_info() if hasattr(self.policy, "get_info") else {}
            )

            # For NFSP, get per-player policy mode
            if "policies" in policy_info and player_id:
                policy_mode = policy_info["policies"].get(
                    player_id, policy_info.get("policy")
                )
            else:
                policy_mode = policy_info.get("policy")

            # Extract reward for the player who took the action
            if isinstance(reward_data, dict):
                # Multi-player: get reward for the acting player
                reward = reward_data.get(player_id, 0.0)
                # Store ALL rewards in info for trainer-side reward accumulation
                if next_info is None:
                    next_info = {}
                next_info["all_player_rewards"] = reward_data
            else:
                reward = reward_data

            game.append(
                observation=next_state,
                info=next_info,
                action=action_val,
                reward=reward,
                policy=policy_mode,
                value=policy_info.get("value"),
                player_id=player_id,
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
