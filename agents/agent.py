import gc
import os
import copy
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import dill as pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import pettingzoo

# Assuming these are custom modules provided in your project structure
from agents.random import RandomAgent
from stats.stats import StatTracker
from agent_configs.base_config import Config
from wrappers import record_video_wrapper, EpisodeTrigger


class BaseAgent(ABC):
    """
    Base Agent class handling generic RL training loops, checkpointing, and testing.
    """

    def __init__(
        self,
        env: gym.Env,
        config: Config,
        name: str,
        device: Optional[torch.device] = None,
        from_checkpoint: bool = False,
    ):
        self.from_checkpoint = from_checkpoint
        self.model_name = name
        self.config = config

        # 1. Device Setup
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        print(f"[{self.model_name}] Using device: {self.device}")

        # 2. Placeholders for Child Classes
        self.model: nn.Module = None
        self.optimizer: optim.Optimizer = None
        self.lr_scheduler: optim.lr_scheduler.LRScheduler = None
        self.replay_buffer = None  # Placeholder

        # 3. Training Params
        # Dynamically determine player_id (PettingZoo uses player_1 for TicTacToe, etc.)
        if hasattr(env, "possible_agents") and len(env.possible_agents) > 0:
            self.player_id = env.possible_agents[0]
        elif hasattr(env, "agents") and len(env.agents) > 0:
            self.player_id = env.agents[0]
        else:
            self.player_id = "player_0"

        self.training_step = 0
        # Safety checks for config values
        total_steps = self.config.training_steps
        self.checkpoint_interval = max(total_steps // 30, 1)
        self.test_interval = max(total_steps // 30, 1)
        self.test_trials = 5

        # 4. Environment Setup
        self.env = env
        self.observation_dimensions, self.observation_dtype = (
            self.determine_observation_dimensions(env)
        )
        print(f"Observation dimensions: {self.observation_dimensions}")

        # 5. Action Space Setup
        self._setup_action_space(env)

        # 6. Test Environment
        self.test_env = self.make_test_env(env)

    def _setup_action_space(self, env):
        """Determines action space properties."""
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.num_actions = int(env.action_space.n)
            self.discrete_action_space = True
        elif callable(env.action_space):  # PettingZoo
            self.num_actions = int(env.action_space(self.player_id).n)
            self.discrete_action_space = True
        else:  # Box/Continuous
            self.num_actions = int(env.action_space.shape[0])
            self.discrete_action_space = False

        print(
            f"Num actions: {self.num_actions} (Discrete: {self.discrete_action_space})"
        )

    def make_test_env(self, env: gym.Env):
        """Creates a separate environment for testing, handling video recording if applicable."""
        print("Making test env...")
        try:
            # Deepcopy can fail on certain C-based envs (MuJoCo, Atari)
            # Ideally, pass a factory function instead of an env instance to __init__
            env_copy = copy.deepcopy(env)
        except Exception as e:
            print(
                f"Warning: Could not deepcopy environment ({e}). Reusing reference (unsafe for concurrent access) or recreate manually."
            )
            env_copy = env

        render_mode = getattr(env, "render_mode", None)

        if render_mode == "rgb_array":
            print("Test env configured for video recording.")
            video_path = str(Path("checkpoints", self.model_name, "videos"))

            if isinstance(env, gym.Env):
                return gym.wrappers.RecordVideo(
                    env_copy,
                    video_folder=video_path,
                    name_prefix=f"{self.model_name}",
                    disable_logger=True,
                )
            elif isinstance(env, (pettingzoo.AECEnv, pettingzoo.ParallelEnv)):
                return record_video_wrapper(env_copy, video_path)

        return env_copy

    def determine_observation_dimensions(self, env):
        """Infers input dimensions for the neural network."""
        obs_space = env.observation_space

        if isinstance(obs_space, gym.spaces.Box):
            return torch.Size(obs_space.shape), obs_space.dtype
        elif isinstance(obs_space, gym.spaces.Discrete):
            return torch.Size((1,)), np.int32
        elif isinstance(obs_space, gym.spaces.Tuple):
            return torch.Size((len(obs_space.spaces),)), np.int32
        elif callable(obs_space):
            # For PettingZoo-style callable observation spaces
            player_id = getattr(self, "player_id", None)
            if player_id is None:
                # If player_id isn't set yet during init, try to find a valid one
                if hasattr(env, "possible_agents") and env.possible_agents:
                    player_id = env.possible_agents[0]
                elif hasattr(env, "agents") and env.agents:
                    player_id = env.agents[0]
                else:
                    # Fallback to a common default
                    player_id = "player_0"

            # Use try-except as a last resort because some wrappers/envs might behave unexpectedly
            try:
                space = obs_space(player_id)
            except KeyError:
                # If the chosen player_id failed, try once with the first possible agent directly
                if hasattr(env, "possible_agents") and env.possible_agents:
                    space = obs_space(env.possible_agents[0])
                else:
                    raise

            return torch.Size(space.shape), space.dtype
        else:
            return torch.Size(obs_space.shape), obs_space.dtype

    def preprocess(self, states) -> torch.Tensor:
        """
        Converts states to torch tensors on the correct device.
        Adds batch dimension if input is a single observation.
        """
        # 1. Convert to numpy (efficiently)
        if torch.is_tensor(states):
            states = states.cpu().numpy()

        np_states = np.array(states, copy=False)

        # 2. Convert to Tensor
        prepared_state = torch.tensor(
            np_states, dtype=torch.float32, device=self.device
        )

        # 3. Handle Batch Dimensions
        # If it's a scalar (0-dim tensor), make it 1D
        if prepared_state.ndim == 0:
            prepared_state = prepared_state.unsqueeze(0)

        # If the shape matches the single observation shape exactly, implies missing batch dim
        # e.g., Obs is (4,), Tensor is (4,) -> make it (1, 4)
        if prepared_state.shape == torch.Size(self.observation_dimensions):
            prepared_state = prepared_state.unsqueeze(0)

        return prepared_state

    # --- Abstract Methods ---
    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def select_actions(self, prediction, info) -> torch.Tensor:
        raise NotImplementedError

    def learn(self):
        pass

    def update_replay_priorities(self, indices, priorities, ids):
        self.replay_buffer.update_priorities(indices, priorities, ids=ids)

    def update_target_model(self):
        """Copy parameters from the learning model to the target model.
        This uses load_state_dict under torch.no_grad() for safety and speed.
        """
        with torch.no_grad():

            if self.config.soft_update:
                for wt, wp in zip(
                    self.target_model.parameters(), self.model.parameters()
                ):
                    wt.copy_(
                        self.config.ema_beta * wt + (1 - self.config.ema_beta) * wp
                    )
            else:
                self.target_model.load_state_dict(self.model.state_dict())

            # If using multiprocessing, ensure target remains in shared memory.
            if self.config.multi_process:
                self.target_model.share_memory()

    # --- Checkpointing ---
    def load_optimizer_state(self, checkpoint):
        if self.optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.lr_scheduler and "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    def load_replay_buffers(self, checkpoint):
        if "replay_buffer" in checkpoint:
            self.replay_buffer = checkpoint["replay_buffer"]

    def load_model_weights(self, checkpoint):
        if self.model and "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])

    def make_checkpoint_dict(self):
        checkpoint = {
            "training_step": self.training_step,
            "model_name": self.model_name,
        }
        if self.optimizer:
            checkpoint["optimizer"] = self.optimizer.state_dict()
        if self.lr_scheduler:
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()
        if self.replay_buffer:
            checkpoint["replay_buffer"] = self.replay_buffer
        if self.model:
            checkpoint["model"] = self.model.state_dict()

        # NOTE: We specifically DO NOT save the environment.
        # It is not picklable in most robust use cases.
        return checkpoint

    def save_checkpoint(self, save_weights=False):
        base_dir = Path("checkpoints", self.model_name)
        step_dir = base_dir / f"step_{self.training_step}"
        os.makedirs(step_dir, exist_ok=True)

        if save_weights:
            weights_dir = step_dir / "model_weights"
            os.makedirs(weights_dir, exist_ok=True)
            weights_path = weights_dir / "weights.pt"
            checkpoint = self.make_checkpoint_dict()
            torch.save(checkpoint, weights_path)

        # Save Config
        config_dir = base_dir / "configs"
        os.makedirs(config_dir, exist_ok=True)
        if hasattr(self.config, "dump"):
            self.config.dump(f"{config_dir}/config.yaml")

        # Save Stats
        stats_dir = step_dir / "graphs_stats"
        os.makedirs(stats_dir, exist_ok=True)

        if hasattr(self, "stats"):
            with open(stats_dir / "stats.pkl", "wb") as f:
                pickle.dump(self.stats.get_data(), f)

            # Plot graphs
            graph_dir = base_dir / "graphs"
            os.makedirs(graph_dir, exist_ok=True)
            self.stats.plot_graphs(dir=graph_dir)

        gc.collect()

    @classmethod
    def load(cls, *args, **kwargs):
        """Alias for load_from_checkpoint."""
        return cls.load_from_checkpoint(*args, **kwargs)

    @classmethod
    def load_from_checkpoint(
        cls, env, agent_class, config_class, dir_path: str, training_step, device
    ):
        """
        Loads an agent from a checkpoint.
        IMPORTANT: 'env' must be passed fresh, it is not loaded from disk.
        """
        dir_path = Path(dir_path)
        step_dir = dir_path / f"step_{training_step}"
        weights_path = (
            step_dir / "model_weights/weights.pt"
        )  # Standardized extension to .pt
        config_path = dir_path / "configs/config.yaml"

        # 1. Load Config
        if not config_path.exists():
            print(
                f"Config not found at {config_path}, attempting to find in step dir..."
            )
            # Fallback logic if needed
        config = config_class.load(config_path)

        # 2. Load Checkpoint Data
        # weights_only=False is needed because we might be loading the replay buffer (arbitrary object)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        model_name = checkpoint.get("model_name", "unnamed_model")

        # 3. Instantiate Agent
        agent = agent_class(
            env=env, config=config, name=model_name, device=device, from_checkpoint=True
        )

        agent.training_step = checkpoint.get("training_step", 0)
        agent.load_model_weights(checkpoint)
        agent.load_optimizer_state(checkpoint)
        agent.load_replay_buffers(checkpoint)

        # 4. Load Stats
        stats_path = step_dir / "graphs_stats/stats.pkl"
        if stats_path.exists():
            with open(stats_path, "rb") as f:
                agent.stats = pickle.load(f)
        else:
            print("Warning: No stats file found for checkpoint.")

        return agent

    def test(self, num_trials, dir="./checkpoints") -> None:
        if num_trials == 0:
            return
        with torch.no_grad():
            """Test the agent."""
            average_score = 0
            max_score = float("-inf")
            min_score = float("inf")
            # self.test_env.reset()
            if self.test_env.render_mode == "rgb_array":
                self.test_env.episode_trigger = EpisodeTrigger(num_trials)
                self.test_env.video_folder = "{}/videos/{}".format(dir, self.model_name)
                if not os.path.exists(self.test_env.video_folder):
                    os.makedirs(self.test_env.video_folder)
            for trials in range(num_trials):
                state, info = self.test_env.reset()

                done = False
                score = 0

                while not done:
                    prediction = self.predict(state, info)
                    action = self.select_actions(prediction, info=info).item()
                    state, reward, terminated, truncated, info = self.test_env.step(
                        action
                    )
                    # self.test_env.render()
                    done = terminated or truncated
                    score += reward[0] if isinstance(reward, list) else reward
                average_score += score
                max_score = max(max_score, score)
                min_score = min(min_score, score)
                print("score: ", score)

            # reset
            # if self.test_env.render_mode != "rgb_array":
            #     self.test_env.render()
            self.test_env.close()
            average_score /= num_trials
            return {
                "score": average_score,
                "max_score": max_score,
                "min_score": min_score,
            }

    def run_tests(self, stats):
        dir = Path("checkpoints", self.model_name)
        training_step_dir = Path(dir, f"step_{self.training_step}")

        test_score = self.test(self.test_trials, dir=training_step_dir)
        print("Test score", test_score)
        if isinstance(test_score, float):
            stats.append("test_score", test_score)
        elif isinstance(test_score, dict):
            for key in test_score:
                stats.append("test_score", test_score[key], subkey=key)
        else:
            print(test_score)
            raise ValueError

    def __getstate__(self):
        # Always work on a copy to avoid mutating the original instance
        try:
            state = super().__getstate__()
            state = state.copy()  # Ensure we have our own copy if super returned a ref
        except AttributeError:
            state = self.__dict__.copy()

        # 2. BaseAgent specific exclusions
        if "env" in state:
            del state["env"]
        if "test_env" in state:
            del state["test_env"]
        if "stats" in state:
            del state["stats"]
        return state


class MARLBaseAgent(BaseAgent):
    def __init__(
        self,
        env,
        config: Config,  # Fixed type hint and argument order logic
        name: str,
        test_agents: List[Any] = None,
        device: Optional[torch.device] = None,
        from_checkpoint=False,
    ):
        if test_agents is None:
            test_agents = [RandomAgent()]

        # FIX: BaseAgent init expects (env, config, name...)
        # The previous code passed 'model' which likely didn't exist or was in wrong place
        super().__init__(
            env, config, name, device=device, from_checkpoint=from_checkpoint
        )
        self.test_agents = test_agents
        print(
            f"MARL Agent '{self.model_name}' initialized. Test agents: {[a.model_name for a in self.test_agents]}"
        )

    def __getstate__(self):
        try:
            state = super().__getstate__()
            state = state.copy()
        except AttributeError:
            state = self.__dict__.copy()

        return state

    def test(self, num_trials, player=0, dir="./checkpoints") -> None:
        if self.config.game.num_players == 1:
            return super(MARLBaseAgent, self).test(num_trials, dir)
        if num_trials == 0:
            return
        with torch.no_grad():
            """Test the agent."""
            results = []
            max_score = float("-inf")
            min_score = float("inf")
            # self.test_env.reset()
            if self.test_env.render_mode == "rgb_array":
                self.test_env.episode_trigger = lambda x: (x + 1) % num_trials == 0
                self.test_env.video_folder = "{}/videos/{}".format(dir, self.model_name)
                if not os.path.exists(self.test_env.video_folder):
                    os.makedirs(self.test_env.video_folder)
            for trials in range(num_trials):
                self.test_env.reset()
                state, reward, terminated, truncated, info = self.test_env.last()
                agent_id = self.test_env.agent_selection
                current_player = self.test_env.agents.index(agent_id)
                # state, info = process_petting_zoo_obs(state, info, current_player)
                done = terminated or truncated
                score = 0

                while not done:
                    prediction = self.predict(state, info, env=self.test_env.env)
                    action = self.select_actions(prediction, info=info).item()

                    self.test_env.step(action)
                    state, reward, terminated, truncated, info = self.test_env.last()
                    agent_id = self.test_env.agent_selection
                    current_player = self.test_env.agents.index(agent_id)
                    # state, info = process_petting_zoo_obs(state, info, current_player)
                    done = terminated or truncated

                    if done:
                        score += self.test_env.rewards[f"player_{player}"]
                results.append(score)

            # reset
            # if self.test_env.render_mode != "rgb_array":
            #     self.test_env.render()
            self.test_env.close()
            average_score = sum(results) / num_trials
            max_score = max(results)
            min_score = min(results)
            # std = np.std(results)

            print("average score:", average_score)
            return {
                "score": average_score,
                "max_score": max_score,
                "min_score": min_score,
                # "std": std,
            }

    def test_vs_agent(self, num_trials, agent, dir="./checkpoints"):
        """
        Test the trained NFSP agent against a random agent
        """
        final_rewards = {player: [] for player in range(self.config.game.num_players)}
        results = {}
        for player in range(self.config.game.num_players):
            print("Testing Player {} vs Agent {}".format(player, agent.model_name))
            if self.test_env.render_mode == "rgb_array":
                self.test_env.episode_trigger = lambda x: (x + 1) % num_trials == 0
                self.test_env.video_folder = "{}/videos/{}".format(
                    dir, agent.model_name
                )
                if not os.path.exists(self.test_env.video_folder):
                    os.makedirs(self.test_env.video_folder)

            with torch.no_grad():  # No gradient computation during testing
                for trial in range(num_trials // self.config.game.num_players):
                    # Reset environment
                    self.test_env.reset()
                    state, reward, termination, truncation, info = self.test_env.last()
                    done = termination or truncation
                    agent_id = self.test_env.agent_selection
                    current_player = self.test_env.agents.index(agent_id)
                    # state, info = process_petting_zoo_obs(state, info, current_player)
                    agent_names = self.test_env.agents.copy()

                    episode_length = 0
                    while not done and episode_length < 1000:  # Safety limit
                        # Get current agent and player
                        episode_length += 1

                        # Get action from average strategy
                        if current_player == player:
                            prediction = self.predict(
                                state, info, env=self.test_env.env
                            )
                            action = self.select_actions(prediction, info=info).item()
                            if trial == 0:
                                print(
                                    f"Player {current_player} prediction: {prediction}"
                                )
                                print(f"action: {action}")

                        else:

                            prediction = agent.predict(
                                state, info, env=self.test_env.env
                            )
                            action = agent.select_actions(prediction, info=info).item()

                            if trial == 0:
                                print(
                                    f"Player {current_player} {agent.model_name} action: {action}"
                                )

                        # Step environment
                        self.test_env.step(action)
                        state, reward, termination, truncation, info = (
                            self.test_env.last()
                        )
                        agent_id = self.test_env.agent_selection
                        current_player = self.test_env.agents.index(agent_id)
                        # state, info = process_petting_zoo_obs(
                        #     state, info, current_player
                        # )
                        done = termination or truncation

                    final_rewards[player].append(
                        self.test_env.rewards[self.test_env.agents[player]]
                    )

            test_player_average_score = sum(final_rewards[player]) / len(
                final_rewards[player]
            )
            test_player_win_percentage = sum(
                1 for r in final_rewards[player] if r > 0
            ) / len(final_rewards[player])
            # std = np.std(final_rewards[player])
            results.update(
                {
                    "player_{}_score".format(player): test_player_average_score,
                }
            )
            results.update(
                {
                    "player_{}_win%".format(player): test_player_win_percentage,
                }
            )
            print(
                f"Player {player} win percentage vs {agent.model_name}: {test_player_win_percentage * 100} and average score: {test_player_average_score}"
            )
        results.update(
            {
                "score": sum(
                    results["player_{}_score".format(player)]
                    for player in range(self.config.game.num_players)
                )
                / self.config.game.num_players
            }
        )
        return results

    def run_tests(self, stats: StatTracker):
        dir = Path("checkpoints", self.model_name)
        training_step_dir = Path(dir, f"step_{self.training_step}")

        for test_agent in self.test_agents:
            results = self.test_vs_agent(
                self.test_trials,
                test_agent,
                dir=training_step_dir,
            )
            print("Results vs {}: {}".format(test_agent.model_name, results))

            for key in results:
                stats.append(
                    "test_score_vs_{}".format(test_agent.model_name),
                    results[key],
                    subkey=key,
                )
        super().run_tests(stats)
