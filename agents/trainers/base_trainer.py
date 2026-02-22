import os
import gc
import torch
import numpy as np
import dill as pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
import gymnasium as gym
from stats.stats import StatTracker


class BaseTrainer:
    """
    Base trainer class containing shared logic for all specialized trainers.
    Handles environment specification inference, stat setup, checkpointing, and testing.
    """

    def __init__(
        self,
        config: Any,
        env: Any,
        device: torch.device,
        model_name: str = "agent",
        stats: Optional[StatTracker] = None,
        test_agents: Optional[List] = None,
    ):
        self.config = config
        self.device = device
        self.model_name = model_name
        self.stats = (
            stats if stats is not None else StatTracker(model_name=model_name)
        )
        self.test_agents = test_agents if test_agents is not None else []
        self._env = env

        # Detect player_id for PettingZoo environments
        self._player_id = self._detect_player_id(env)

        # Get observation/action specs
        self.obs_dim, self.obs_dtype = self._determine_observation_dimensions(env)
        self.num_actions = self._get_num_actions(env)
        self.num_players = self._get_num_players(env)

        self.training_step = 0

        # Intervals
        total_steps = config.training_steps
        self.checkpoint_interval = max(total_steps // 30, 1)
        self.test_interval = max(total_steps // 30, 1)
        self.test_trials = 5

    def _detect_player_id(self, env) -> str:
        """Detects the player_id for PettingZoo environments."""
        if hasattr(env, "possible_agents") and len(env.possible_agents) > 0:
            return env.possible_agents[0]
        elif hasattr(env, "agents") and len(env.agents) > 0:
            return env.agents[0]
        else:
            return "player_0"

    def _determine_observation_dimensions(self, env) -> Tuple[torch.Size, np.dtype]:
        """
        Infers input dimensions for the neural network.
        Ported from BaseAgent.determine_observation_dimensions.
        """
        obs_space = env.observation_space

        if isinstance(obs_space, gym.spaces.Box):
            return torch.Size(obs_space.shape), obs_space.dtype
        elif isinstance(obs_space, gym.spaces.Discrete):
            return torch.Size((1,)), np.int32
        elif isinstance(obs_space, gym.spaces.Tuple):
            return torch.Size((len(obs_space.spaces),)), np.int32
        elif callable(obs_space):
            # For PettingZoo-style callable observation spaces
            player_id = getattr(self, "_player_id", "player_0")
            try:
                space = obs_space(player_id)
            except KeyError:
                if hasattr(env, "possible_agents") and env.possible_agents:
                    space = obs_space(env.possible_agents[0])
                else:
                    raise
            return torch.Size(space.shape), space.dtype
        else:
            return torch.Size(obs_space.shape), obs_space.dtype

    def _get_num_actions(self, env) -> int:
        """
        Determines action space properties.
        Ported from BaseAgent._setup_action_space.
        """
        if isinstance(env.action_space, gym.spaces.Discrete):
            return int(env.action_space.n)
        elif callable(env.action_space):  # PettingZoo
            player_id = getattr(self, "_player_id", "player_0")
            return int(env.action_space(player_id).n)
        elif self.config.game.num_actions:
            return self.config.game.num_actions
        else:
            # Box/Continuous
            return int(env.action_space.shape[0])

    def _get_num_players(self, env) -> int:
        """Determines the number of players in the environment."""
        if hasattr(env, "possible_agents"):
            return len(env.possible_agents)
        elif hasattr(env, "agents"):
            return len(env.agents)
        elif self.config.game.num_players:
            return self.config.game.num_players
        return 1

    def _save_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """
        Saves model weights and stats. Standardized checkpoint format.
        """
        base_dir = Path("checkpoints", self.model_name)
        step_dir = base_dir / f"step_{self.training_step}"
        os.makedirs(step_dir, exist_ok=True)

        # Save weights
        weights_dir = step_dir / "model_weights"
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = weights_dir / "weights.pt"

        full_checkpoint = {
            "training_step": self.training_step,
            "model_name": self.model_name,
            **checkpoint_data,
        }
        torch.save(full_checkpoint, weights_path)

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
        abs_path = os.path.abspath(step_dir)
        print(f"Saved checkpoint at step {self.training_step} to {abs_path}")

    @classmethod
    def load_from_checkpoint(
        cls,
        env: Any,
        config_class: Type,
        dir_path: str,
        training_step: int,
        device: torch.device,
        **extra_init_kwargs,
    ):
        """
        Standardized loading logic for all trainers.
        """
        dir_path = Path(dir_path)
        step_dir = dir_path / f"step_{training_step}"
        weights_path = step_dir / "model_weights/weights.pt"
        config_path = dir_path / "configs/config.yaml"

        # 1. Load Config
        config = config_class.load(config_path)

        # 2. Load Checkpoint Data (weights_only=False for complex objects if needed)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        # 3. Instantiate Trainer
        trainer = cls(config=config, env=env, device=device, **extra_init_kwargs)
        trainer.training_step = checkpoint.get("training_step", 0)

        # 4. Load weights - this needs to be implemented by child classes
        trainer.load_checkpoint_weights(checkpoint)

        # 5. Load Stats
        stats_path = step_dir / "graphs_stats/stats.pkl"
        if stats_path.exists():
            with open(stats_path, "rb") as f:
                trainer.stats.set_data(pickle.load(f))

        return trainer

    def load_checkpoint_weights(self, checkpoint: Dict[str, Any]):
        """Child classes should implement this to load specific weights from checkpoint dictionary."""
        raise NotImplementedError

    def test(self, num_trials: int, dir: str = "./checkpoints") -> Dict[str, float]:
        """
        Runs evaluation episodes vs self.
        Handles both Gymnasium (single-player) and PettingZoo (multi-player) environments.
        """
        if num_trials == 0:
            return {}

        test_env = self.config.game.make_env()
        scores = []
        is_multiplayer = self.config.game.num_players > 1

        with torch.inference_mode():
            for _ in range(num_trials):
                if is_multiplayer:
                    test_env.reset()
                    state, reward, terminated, truncated, info = test_env.last()
                else:
                    state, info = test_env.reset()

                done = False
                episode_reward = 0.0
                episode_length = 0

                while not done and episode_length < 1000:
                    episode_length += 1
                    action = self.select_test_action(state, info, test_env)
                    action_val = action.item() if hasattr(action, "item") else action

                    if is_multiplayer:
                        test_env.step(action_val)
                        state, reward, terminated, truncated, info = test_env.last()
                        done = terminated or truncated
                        if done:
                            # For evaluation, we often track the first player's reward or a sum
                            # Here we use the PettingZoo rewards dict for the first agent
                            first_agent = test_env.possible_agents[0]
                            episode_reward = float(
                                test_env.rewards.get(first_agent, 0.0)
                            )
                    else:
                        state, reward, terminated, truncated, info = test_env.step(
                            action_val
                        )
                        episode_reward += float(reward)
                        done = terminated or truncated

                scores.append(episode_reward)

        test_env.close()

        if not scores:
            return {}

        return {
            "score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
        }

    def test_vs_agent(self, num_trials: int, agent: Any, dir: str = "./checkpoints"):
        """Test the trained agent against another agent (multi-player)."""
        test_env = self.config.game.make_env()
        num_players = self.config.game.num_players
        final_rewards = {player: [] for player in range(num_players)}
        results = {}

        with torch.no_grad():
            for player in range(num_players):
                print(f"Testing Player {player} vs Agent {agent.model_name}")

                for trial in range(num_trials // num_players):
                    test_env.reset()
                    state, reward, termination, truncation, info = test_env.last()
                    done = termination or truncation

                    episode_length = 0
                    while not done and episode_length < 1000:
                        episode_length += 1
                        agent_id = test_env.agent_selection
                        current_player_idx = test_env.agents.index(agent_id)

                        if current_player_idx == player:
                            # Our agent's turn
                            action = self.select_test_action(state, info, test_env)
                        else:
                            # Opponent's turn
                            prediction = agent.predict(state, info, env=test_env)
                            action = agent.select_actions(prediction, info=info)

                        # Ensure action is a standard scalar for PettingZoo/Gymnasium
                        action_val = (
                            action.item() if hasattr(action, "item") else action
                        )
                        test_env.step(action_val)
                        state, reward, termination, truncation, info = test_env.last()
                        done = termination or truncation

                    final_rewards[player].append(
                        test_env.rewards[test_env.agents[player]]
                    )

                avg_score = sum(final_rewards[player]) / len(final_rewards[player])
                win_pct = sum(1 for r in final_rewards[player] if r > 0) / len(
                    final_rewards[player]
                )
                results[f"player_{player}_score"] = avg_score
                results[f"player_{player}_win%"] = win_pct
                print(
                    f"Player {player} win%: {win_pct*100:.1f}%, avg score: {avg_score:.2f}"
                )

        player_scores = [results[f"player_{p}_score"] for p in range(num_players)]
        results["score"] = sum(player_scores) / num_players
        results["min_score"] = min(player_scores)
        results["max_score"] = max(player_scores)

        test_env.close()
        return results

    def _run_tests(self) -> None:
        """Runs test episodes vs self and potentially vs other agents."""
        # 1. Self-test
        test_results = self.test(self.test_trials)
        if test_results:
            self.stats.append("test_score", test_results.get("score"), subkey="avg")
            self.stats.append("test_score", test_results.get("min_score"), subkey="min")
            self.stats.append("test_score", test_results.get("max_score"), subkey="max")
            # Log player-specific scores if available (mostly for multiplayer)
            for p in range(self.config.game.num_players):
                player_score_key = f"player_{p}_score"
                if player_score_key in test_results:
                    self.stats.append(
                        "test_score", test_results[player_score_key], subkey=f"p{p}"
                    )
            print(f"Test score: {test_results.get('score'):.3f}")

        # 2. Test vs agents (if any)
        for agent in self.test_agents:
            vs_results = self.test_vs_agent(self.test_trials, agent)
            key = f"vs_{agent.model_name}_score"
            self.stats.append(key, vs_results["score"], subkey="avg")
            self.stats.append(key, vs_results["min_score"], subkey="min")
            self.stats.append(key, vs_results["max_score"], subkey="max")
            for p in range(self.config.game.num_players):
                player_score_key = f"player_{p}_score"
                if player_score_key in vs_results:
                    self.stats.append(key, vs_results[player_score_key], subkey=f"p{p}")

    def _setup_stats(self):
        """Initializes the stat tracker with common keys and plot types."""
        from stats.stats import PlotType

        stat_keys = [
            "score",
            "episode_length",
            "test_score",
        ]

        # Add test_score vs other agents if applicable
        if hasattr(self, "test_agents") and self.test_agents:
            for agent in self.test_agents:
                stat_keys.append(f"vs_{agent.model_name}_score")

        # Initialize keys
        player_subkeys = [f"p{p}" for p in range(self.config.game.num_players)]
        test_subkeys = ["avg", "min", "max"] + player_subkeys

        for key in stat_keys:
            if key not in self.stats.stats:
                if "test_score" in key:
                    self.stats._init_key(key, subkeys=test_subkeys)
                else:
                    self.stats._init_key(key)

        # Add common plot types
        self.stats.add_plot_types(
            "score",
            PlotType.ROLLING_AVG,
            PlotType.BEST_FIT_LINE,
            rolling_window=100,
            ema_beta=0.6,
        )
        self.stats.add_plot_types(
            "test_score", PlotType.BEST_FIT_LINE, PlotType.VARIATION_FILL
        )
        self.stats.add_plot_types(
            "episode_length", PlotType.ROLLING_AVG, rolling_window=100
        )

        # Test vs agents
        if hasattr(self, "test_agents") and self.test_agents:
            for agent in self.test_agents:
                self.stats.add_plot_types(
                    f"vs_{agent.model_name}_score",
                    PlotType.BEST_FIT_LINE,
                    PlotType.VARIATION_FILL,
                )

    def select_test_action(self, state, info, env) -> Any:
        """Override in child classes to specify how to select an action during testing."""
        raise NotImplementedError
