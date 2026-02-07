import time
import torch
import numpy as np
from typing import Optional, Dict, Any, List
from executors import LocalExecutor, TorchMPExecutor
from agents.muzero_learner import MuZeroLearner
from agents.policies.search_policy import SearchPolicy
from agents.action_selectors.selectors import TemperatureSelector
from search.search_factories import create_mcts
from agents.actors import GenericActor
from modules.agent_nets.muzero import Network
from stats.stats import StatTracker


class MuZeroTrainer:
    """
    MuZeroTrainer orchestrates the training process by coordinating
    the executor for data collection, the replay buffer for storage,
    and the learner for optimization.
    """

    def __init__(
        self,
        config,
        env,  # env is used to get some specs, but not for training directly
        device: torch.device,
        stats: Optional[StatTracker] = None,
        test_agents: List = None,  # Add this parameter
    ):
        self.config = config
        self.device = device
        self.stats = (
            stats if stats is not None else StatTracker(model_name=config.model_name)
        )

        self.test_agents = test_agents if test_agents is not None else []

        # 1. Initialize Network
        # Handle env as factory function or instance
        if callable(env) and not hasattr(env, "observation_space"):
            self._env_factory = env
            env = env()  # Call factory to get instance
        else:
            self._env_factory = self.config.game.make_env

        self._env = env  # Store for later use

        # Detect player_id for PettingZoo environments
        if hasattr(env, "possible_agents") and len(env.possible_agents) > 0:
            self._player_id = env.possible_agents[0]
        elif hasattr(env, "agents") and len(env.agents) > 0:
            self._player_id = env.agents[0]
        else:
            self._player_id = "player_0"

        # Get observation dimensions using BaseAgent-style logic
        obs_dim, obs_dtype = self._determine_observation_dimensions(env)
        obs_dtype = getattr(self.config, "observation_dtype", obs_dtype)

        # Get num_actions using BaseAgent-style logic
        num_actions = self._get_num_actions(env)

        kwargs = {}
        if (
            hasattr(self.config, "world_model_cls")
            and self.config.world_model_cls is not None
        ):
            kwargs["world_model_cls"] = self.config.world_model_cls

        self.model = Network(
            config=config,
            num_actions=num_actions,
            input_shape=torch.Size((self.config.minibatch_size,) + obs_dim),
            channel_first=True,
            **kwargs,
        )
        if self.config.multi_process:
            self.model.share_memory()
        self.model.to(self.device)

        # 2. Create Search Algorithm and Action Selector
        self.search_alg = create_mcts(self.config, self.device, num_actions)
        self.action_selector = TemperatureSelector()

        # 3. Initialize Policy with dependency injection
        self.policy = SearchPolicy(
            model=self.model,
            search_algorithm=self.search_alg,
            action_selector=self.action_selector,
            config=self.config,
            device=self.device,
            observation_dimensions=obs_dim,
        )

        # 3. Initialize Learner (handles buffer and optimizer)
        obs_dtype = getattr(self.config, "observation_dtype", torch.float32)
        self.learner = MuZeroLearner(
            config=self.config,
            model=self.model,
            device=self.device,
            num_actions=num_actions,
            observation_dimensions=obs_dim,
            observation_dtype=obs_dtype,
            policy=self.policy,
        )
        self.buffer = self.learner.replay_buffer

        # 4. Initialize Executor
        if self.config.multi_process:
            self.executor = TorchMPExecutor()
        else:
            self.executor = LocalExecutor()

        # Launch workers
        worker_args = (
            self.config.game.make_env,
            self.policy,
            self.config.game.num_players,
        )
        self.executor.launch(GenericActor, worker_args, self.config.num_workers)

        self.training_step = 0

        # Checkpointing and testing intervals (can be overridden after init)
        total_steps = self.config.training_steps
        self.checkpoint_interval = max(total_steps // 30, 1)
        self.test_interval = max(total_steps // 30, 1)
        self.test_trials = 5
        self.model_name = self.config.model_name

    def train(self):
        """
        Main training loop.
        """
        self._setup_stats()

        print(f"Starting training for {self.config.training_steps} steps...")
        start_time = time.time()

        while self.training_step < self.config.training_steps:
            # 1. Collect data from executor
            # We use collect_data which accumulates until min_samples if needed
            # For MuZero, we might want to collect at least 1 game before learning
            data, collect_stats = self.executor.collect_data(min_samples=1)

            # 2. Store data in buffer
            for game in data:
                self.buffer.store_aggregate(game_object=game)

            # 3. Log collection stats
            for key, val in collect_stats.items():
                self.stats.append(key, val)

            # 4. Learning step
            # Learner.step samples from buffer and performs optimization
            if self.buffer.size >= self.config.min_replay_buffer_size:
                for _ in range(self.config.num_minibatches):
                    loss_stats = self.learner.step(self.stats)
                    if loss_stats:
                        for key, val in loss_stats.items():
                            self.stats.append(key, val)

                self.training_step += 1

                # 5. Update workers (if needed)
                # In TorchMP with shared memory, this might be a no-op if using the same model instance.
                # But we follow the pattern for consistency.
                if self.training_step % self.config.transfer_interval == 0:
                    self.executor.update_weights(self.model.state_dict())

                # 6. Periodic checkpointing
                if self.training_step % self.checkpoint_interval == 0:
                    self._save_checkpoint()

                # 7. Periodic testing
                if self.training_step % self.test_interval == 0:
                    self._run_tests()

            # Periodic logging
            if self.training_step % 100 == 0 and self.training_step > 0:
                print(f"Step {self.training_step}")

        self.executor.stop()
        # Final checkpoint and stats plot
        self._save_checkpoint()
        print("Training finished.")

    def _save_checkpoint(self):
        """Internal method to save checkpoint using BaseAgent-style pattern."""
        from pathlib import Path
        import gc
        import os
        import dill as pickle

        base_dir = Path("checkpoints", self.model_name)
        step_dir = base_dir / f"step_{self.training_step}"
        os.makedirs(step_dir, exist_ok=True)

        # Save weights
        weights_dir = step_dir / "model_weights"
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = weights_dir / "weights.pt"
        checkpoint = {
            "training_step": self.training_step,
            "model_name": self.model_name,
            "model": self.model.state_dict(),
            "optimizer": self.learner.optimizer.state_dict(),
        }
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
        abs_path = os.path.abspath(step_dir)
        print(f"Saved checkpoint at step {self.training_step} to {abs_path}")

    def _run_tests(self):
        """Run tests at checkpoint intervals."""
        from pathlib import Path

        dir = Path("checkpoints", self.model_name)
        training_step_dir = Path(dir, f"step_{self.training_step}")

        # 1. Run self-evaluation (works for single and multi-player)
        test_results = self.test(self.test_trials, dir=training_step_dir)
        if test_results:
            self.stats.append("test_score", test_results.get("score"), subkey="avg")
            self.stats.append("test_score", test_results.get("min_score"), subkey="min")
            self.stats.append("test_score", test_results.get("max_score"), subkey="max")
            print(f"Test score: {test_results.get('score'):.3f}")

        # 2. Run vs test agents (multi-player only)
        if hasattr(self, "test_agents") and self.test_agents:
            for test_agent in self.test_agents:
                results = self.test_vs_agent(
                    self.test_trials,
                    test_agent,
                    dir=training_step_dir,
                )
                print(f"Results vs {test_agent.model_name}: {results}")

                for key in results:
                    if key == "score":
                        # Log consolidated min/avg/max
                        self.stats.append(
                            f"test_score_vs_{test_agent.model_name}",
                            results["score"],
                            subkey="avg",
                        )
                        self.stats.append(
                            f"test_score_vs_{test_agent.model_name}",
                            results.get("min_score", results["score"]),
                            subkey="min",
                        )
                        self.stats.append(
                            f"test_score_vs_{test_agent.model_name}",
                            results.get("max_score", results["score"]),
                            subkey="max",
                        )
                    elif key in ["min_score", "max_score"]:
                        continue  # Already handled in the 'score' block
                    elif "score" in key:
                        # Skip individual player scores for the main plot to keep it clean
                        continue
                    else:
                        # Log other results (like win%) under their own keys or as subkeys
                        self.stats.append(
                            f"test_score_vs_{test_agent.model_name}",
                            results[key],
                            subkey=key,
                        )

    def test(self, num_trials: int, dir="./checkpoints") -> Dict[str, float]:
        """
        Run evaluation episodes and return test scores.

        Args:
            num_trials: Number of evaluation episodes to run.
            dir: Directory for video recording (if enabled).

        Returns:
            Dictionary with 'score', 'max_score', 'min_score' keys.
        """
        if num_trials == 0:
            return {}

        test_env = self._env_factory()
        scores = []
        is_multiplayer = self.config.game.num_players > 1

        with torch.inference_mode():
            for trial in range(num_trials):
                if is_multiplayer:
                    # Multi-player (PettingZoo style)
                    test_env.reset()
                    state, _, terminated, truncated, info = test_env.last()
                    done = terminated or truncated
                else:
                    # Single-player (Gymnasium style)
                    state, info = test_env.reset()
                    done = False

                episode_reward = 0.0
                episode_length = 0

                while not done and episode_length < 1000:
                    episode_length += 1

                    # Use policy to get action
                    action, _ = self.policy.predict(state, info, env=test_env)
                    action_val = action.item() if hasattr(action, "item") else action

                    if is_multiplayer:
                        test_env.step(action_val)
                        state, reward, terminated, truncated, info = test_env.last()
                        done = terminated or truncated
                        if done:
                            # Get final reward for player 0
                            player_id = test_env.possible_agents[0]
                            episode_reward = test_env.rewards.get(player_id, 0.0)
                    else:
                        state, reward, terminated, truncated, info = test_env.step(
                            action_val
                        )
                        episode_reward += reward
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

    def test_vs_agent(self, num_trials, agent, dir="./checkpoints"):
        """Test the trained agent against another agent."""
        import os

        # Create test env
        test_env = self._env_factory()

        final_rewards = {player: [] for player in range(self.config.game.num_players)}
        results = {}

        with torch.no_grad():
            for player in range(self.config.game.num_players):
                print(f"Testing Player {player} vs Agent {agent.model_name}")

                for trial in range(num_trials // self.config.game.num_players):
                    test_env.reset()
                    state, reward, termination, truncation, info = test_env.last()
                    done = termination or truncation
                    agent_id = test_env.agent_selection
                    current_player = test_env.agents.index(agent_id)

                    episode_length = 0
                    while not done and episode_length < 1000:
                        episode_length += 1

                        if current_player == player:
                            # Our agent's turn - use policy
                            action, _ = self.policy.predict(state, info, env=test_env)
                            action = (
                                action.item() if hasattr(action, "item") else action
                            )
                        else:
                            # Opponent's turn
                            prediction = agent.predict(state, info, env=test_env)
                            action = agent.select_actions(prediction, info=info).item()

                        test_env.step(action)
                        state, reward, termination, truncation, info = test_env.last()
                        agent_id = test_env.agent_selection
                        current_player = test_env.agents.index(agent_id)
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

        player_scores = [
            results[f"player_{p}_score"] for p in range(self.config.game.num_players)
        ]
        results["score"] = sum(player_scores) / self.config.game.num_players
        results["min_score"] = min(player_scores)
        results["max_score"] = max(player_scores)

        test_env.close()
        return results

    def _determine_observation_dimensions(self, env):
        """
        Infers input dimensions for the neural network.
        Ported from BaseAgent.determine_observation_dimensions.
        """
        import gymnasium as gym

        obs_space = env.observation_space

        if isinstance(obs_space, gym.spaces.Box):
            return torch.Size(obs_space.shape), obs_space.dtype
        elif isinstance(obs_space, gym.spaces.Discrete):
            return torch.Size((1,)), np.int32
        elif isinstance(obs_space, gym.spaces.Tuple):
            return torch.Size((len(obs_space.spaces),)), np.int32
        elif callable(obs_space):
            # For PettingZoo-style callable observation spaces
            player_id = self._player_id
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

    def _get_num_actions(self, env):
        """
        Determines action space properties.
        Ported from BaseAgent._setup_action_space.
        """
        import gymnasium as gym

        if isinstance(env.action_space, gym.spaces.Discrete):
            return int(env.action_space.n)
        elif callable(env.action_space):  # PettingZoo
            return int(env.action_space(self._player_id).n)
        elif hasattr(self.config.game, "num_actions"):
            return self.config.game.num_actions
        else:
            # Box/Continuous
            return int(env.action_space.shape[0])

    def _setup_stats(self):
        """Initializes the stat tracker with all required keys and plot types."""
        from stats.stats import PlotType

        test_score_keys = (
            [f"test_score_vs_{agent.model_name}" for agent in self.test_agents]
            if hasattr(self, "test_agents")
            else []
        )

        stat_keys = [
            "score",
            "policy_loss",
            "value_loss",
            "reward_loss",
            "to_play_loss",
            "cons_loss",
            "loss",
            "test_score",
            "episode_length",
            "policy_entropy",
            "value_diff",
            "policy_improvement",
            "learner_fps",
            "actor_fps",
        ] + test_score_keys

        # Initialize keys
        for key in stat_keys:
            if key not in self.stats.stats:
                if "test_score" in key:
                    self.stats._init_key(key, subkeys=["avg", "min", "max"])
                else:
                    self.stats._init_key(key)

        # Add plot types
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
        if test_score_keys:
            for key in test_score_keys:
                self.stats.add_plot_types(
                    key, PlotType.BEST_FIT_LINE, PlotType.VARIATION_FILL
                )
        self.stats.add_plot_types(
            "policy_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "value_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "reward_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "to_play_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types("cons_loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types(
            "episode_length", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "policy_entropy", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "value_diff", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types("policy_improvement", PlotType.BAR)
        self.stats.add_plot_types(
            "learner_fps", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types("actor_fps", PlotType.ROLLING_AVG, rolling_window=100)
