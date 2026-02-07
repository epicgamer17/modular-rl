"""
PPOTrainer orchestrates the training process for PPO by coordinating
the executor for data collection, the learner for optimization, and
handling the on-policy training loop.
"""

import time
import gc
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np
import dill as pickle

from executors import LocalExecutor, TorchMPExecutor
from agents.learners.ppo_learner import PPOLearner
from agents.policies.direct_policy import DirectPolicy
from agents.action_selectors.selectors import CategoricalSelector
from agents.actors import GenericActor
from modules.agent_nets.ppo import PPONetwork
from stats.stats import StatTracker, PlotType


class PPOTrainer:
    """
    PPOTrainer orchestrates the training process by coordinating
    the executor for data collection and the learner for optimization.
    """

    def __init__(
        self,
        config,
        env,
        device: torch.device,
        stats: Optional[StatTracker] = None,
        test_agents: Optional[List] = None,
    ):
        """
        Initializes the PPOTrainer.

        Args:
            config: PPOConfig with hyperparameters.
            env: Environment instance or factory function.
            device: Torch device for training.
            stats: Optional StatTracker for logging metrics.
            test_agents: Optional list of agents to test against.
        """
        self.config = config
        self.device = device
        self.stats = (
            stats if stats is not None else StatTracker(model_name=config.model_name)
        )
        self.test_agents = test_agents if test_agents is not None else []

        # Handle env as factory function or instance
        if callable(env) and not hasattr(env, "observation_space"):
            self._env_factory = env
            env = env()
        else:
            self._env_factory = config.game.make_env

        self._env = env

        # Get observation/action specs
        obs_dim, obs_dtype = self._determine_observation_dimensions(env)
        obs_dtype = getattr(config, "observation_dtype", obs_dtype)
        num_actions = self._get_num_actions(env)
        self.num_actions = num_actions

        # 1. Initialize Network
        input_shape = torch.Size((config.minibatch_size,) + obs_dim)
        self.model = PPONetwork(
            config=config,
            output_size=num_actions,
            input_shape=input_shape,
            discrete=True,  # PPO discrete action space
        )
        self.model.to(device)

        # Initialize weights
        if config.kernel_initializer is not None:
            self.model.initialize(config.kernel_initializer)

        if getattr(config, "multi_process", False):
            self.model.share_memory()

        # 2. Initialize Action Selector (Categorical for stochastic policy)
        self.action_selector = CategoricalSelector(from_logits=False)

        # 3. Initialize Policy
        self.policy = PPOPolicy(
            model=self.model,
            action_selector=self.action_selector,
            device=device,
        )

        # 4. Initialize Learner
        self.learner = PPOLearner(
            config=config,
            model=self.model,
            device=device,
            num_actions=num_actions,
            observation_dimensions=obs_dim,
            observation_dtype=obs_dtype,
        )

        # 5. Initialize Executor
        if getattr(config, "multi_process", False):
            self.executor = TorchMPExecutor()
        else:
            self.executor = LocalExecutor()

        # Launch workers (default to 1 worker if not specified)
        num_workers = getattr(config, "num_workers", 1)
        worker_args = (
            config.game.make_env,
            self.policy,
            config.game.num_players,
        )
        self.executor.launch(GenericActor, worker_args, num_workers)

        self.training_step = 0
        self.model_name = config.model_name

        # Intervals
        total_steps = config.training_steps
        self.checkpoint_interval = max(total_steps // 30, 1)
        self.test_interval = max(total_steps // 30, 1)
        self.test_trials = 5

    def train(self) -> None:
        """
        Main training loop.
        """
        self._setup_stats()

        print(f"Starting PPO training for {self.config.training_steps} steps...")
        start_time = time.time()

        # Track residues between epochs for accurate episode-level logging
        current_episode_score = 0.0
        current_episode_length = 0
        completed_scores = []
        completed_lengths = []

        while self.training_step < self.config.training_steps:
            # 1. Broadcast weights to workers
            self.executor.update_weights(self.model.state_dict())

            # 2. Collect trajectory data (steps_per_epoch transitions)
            steps_collected = 0

            while steps_collected < self.config.steps_per_epoch:
                with torch.no_grad():
                    # Get state from environment
                    state, info = self._env.reset()
                    done = False
                    current_episode_score = 0.0
                    current_episode_length = 0

                    while not done and steps_collected < self.config.steps_per_epoch:
                        # Compute action and value
                        action, log_prob, value = self.policy.compute_action_with_info(
                            state, info
                        )
                        action_val = (
                            action.item() if hasattr(action, "item") else action
                        )

                        # Environment step
                        next_state, reward, terminated, truncated, next_info = (
                            self._env.step(action_val)
                        )
                        done = terminated or truncated

                        # Store transition
                        self.learner.store(
                            observation=state,
                            action=action_val,
                            value=value,
                            log_probability=log_prob,
                            reward=reward,
                        )

                        state = next_state
                        info = next_info
                        current_episode_score += reward
                        current_episode_length += 1
                        steps_collected += 1

                        if done:
                            completed_scores.append(current_episode_score)
                            completed_lengths.append(current_episode_length)
                            # reset for next episode in same epoch
                            current_episode_score = 0.0
                            current_episode_length = 0

                    # Finish trajectory with bootstrap value
                    if done:
                        last_value = 0.0
                    else:
                        with torch.inference_mode():
                            obs = self.learner.preprocess(state)
                            last_value = self.model.critic(obs).item()

                    self.learner.finish_trajectory(last_value)

            # Log collection stats
            if completed_scores:
                for s, l in zip(completed_scores, completed_lengths):
                    self.stats.append("score", float(s))
                    self.stats.append("episode_length", float(l))

                # Print diagnostics
                if self.training_step % 10 == 0:
                    avg_score = float(np.mean(completed_scores))
                    print(
                        f"Step {self.training_step}, "
                        f"Avg Score: {avg_score:.2f}, "
                        f"Episodes Finished: {len(completed_scores)}"
                    )

                # Clear for next stats reporting window
                completed_scores = []
                completed_lengths = []
            else:
                avg_score = 0.0

            # 3. Learning step
            loss_stats = self.learner.step(self.stats)
            if loss_stats:
                for key, val in loss_stats.items():
                    self.stats.append(key, val)

            self.training_step += 1

            # 4. Periodic checkpointing
            if self.training_step % self.checkpoint_interval == 0:
                self._save_checkpoint()

            # 5. Periodic testing
            if self.training_step % self.test_interval == 0:
                self._run_tests()

            # Periodic logging
            if self.training_step % 10 == 0:
                print(
                    f"Step {self.training_step}, "
                    f"Avg Score: {avg_score:.2f}, "
                    f"Episodes Finished: {len(completed_scores)}"
                )

        self.executor.stop()
        self._save_checkpoint()
        print("Training finished.")

    def _run_tests(self) -> None:
        """
        Runs test episodes and logs results.
        """
        dir = Path("checkpoints", self.model_name)
        step_dir = Path(dir, f"step_{self.training_step}")

        test_results = self.test(self.test_trials, dir=step_dir)
        if test_results:
            self.stats.append("test_score", test_results.get("score"))
            print(f"Test score: {test_results.get('score'):.3f}")

    def test(self, num_trials: int, dir: str = "./checkpoints") -> Dict[str, float]:
        """
        Runs evaluation episodes and returns test scores.

        Args:
            num_trials: Number of evaluation episodes.
            dir: Directory for saving results.

        Returns:
            Dictionary with 'score', 'max_score', 'min_score' keys.
        """
        if num_trials == 0:
            return {}

        test_env = self._env_factory()
        scores = []

        with torch.inference_mode():
            for _ in range(num_trials):
                state, info = test_env.reset()
                done = False
                episode_reward = 0.0
                episode_length = 0

                while not done and episode_length < 1000:
                    episode_length += 1

                    # Use policy (greedy for testing)
                    action = self.policy.compute_action(state, info)
                    action_val = action.item() if hasattr(action, "item") else action

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

    def _save_checkpoint(self) -> None:
        """
        Saves model weights and stats.
        """
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
            "actor_optimizer": self.learner.actor_optimizer.state_dict(),
            "critic_optimizer": self.learner.critic_optimizer.state_dict(),
            "actor_scheduler": self.learner.actor_scheduler.state_dict(),
            "critic_scheduler": self.learner.critic_scheduler.state_dict(),
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

    def _determine_observation_dimensions(self, env):
        """
        Infers input dimensions for the neural network.
        """
        import gymnasium as gym

        obs_space = env.observation_space

        if isinstance(obs_space, gym.spaces.Box):
            return torch.Size(obs_space.shape), obs_space.dtype
        elif isinstance(obs_space, gym.spaces.Discrete):
            return torch.Size((1,)), np.int32
        elif isinstance(obs_space, gym.spaces.Tuple):
            return torch.Size((len(obs_space.spaces),)), np.int32
        else:
            return torch.Size(obs_space.shape), obs_space.dtype

    def _get_num_actions(self, env) -> int:
        """
        Determines the number of discrete actions.
        """
        import gymnasium as gym

        if isinstance(env.action_space, gym.spaces.Discrete):
            return int(env.action_space.n)
        elif hasattr(self.config.game, "num_actions"):
            return self.config.game.num_actions
        else:
            return int(env.action_space.shape[0])

    def _setup_stats(self) -> None:
        """
        Initializes the stat tracker with all required keys and plot types.
        """
        stat_keys = [
            "score",
            "actor_loss",
            "critic_loss",
            "kl_divergence",
            "test_score",
            "episode_length",
            "learner_fps",
        ]

        for key in stat_keys:
            if key not in self.stats.stats:
                self.stats._init_key(key)

        self.stats.add_plot_types(
            "score",
            PlotType.ROLLING_AVG,
            PlotType.BEST_FIT_LINE,
            rolling_window=100,
        )
        self.stats.add_plot_types(
            "test_score",
            PlotType.BEST_FIT_LINE,
            PlotType.ROLLING_AVG,
            rolling_window=100,
        )
        self.stats.add_plot_types(
            "actor_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "critic_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "kl_divergence", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "episode_length", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "learner_fps", PlotType.ROLLING_AVG, rolling_window=100
        )


class PPOPolicy(DirectPolicy):
    """
    Extended DirectPolicy for PPO that returns action, log_prob, and value.
    """

    def compute_action(
        self, obs: Any, info: Dict[str, Any] = None, exploration: bool = False
    ) -> Any:
        """
        Computes an action given an observation and info.
        For PPO, we only use the actor output from the model.
        """
        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.tensor(
                obs, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
        else:
            obs_tensor = obs.to(self.device)
            if obs_tensor.dim() == len(self.model.actor.input_shape) - 1:
                obs_tensor = obs_tensor.unsqueeze(0)

        with torch.inference_mode():
            # PPONetwork returns (actor_logits, critic_value)
            logits, _ = self.model(obs_tensor)

        action = self.action_selector.select(logits, exploration=exploration, info=info)

        if action.dim() > 0 and action.shape[0] == 1:
            action = action.squeeze(0)

        return action

    def compute_action_with_info(
        self, obs: Any, info: Dict[str, Any] = None
    ) -> tuple[torch.Tensor, float, float]:
        """
        Computes action, log probability, and value for PPO training.

        Returns:
            Tuple of (action, log_probability, value).
        """
        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.tensor(
                obs, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
        else:
            obs_tensor = obs.to(self.device)
            if obs_tensor.dim() == len(self.model.actor.input_shape) - 1:
                obs_tensor = obs_tensor.unsqueeze(0)

        with torch.inference_mode():
            # Get actor probs and critic value
            probs = self.model.actor(obs_tensor)
            value = self.model.critic(obs_tensor)

        # Use the selector to get the action (sampling mode)
        # Note: action_selector is configured with from_logits=False in PPOTrainer
        action = self.action_selector.select(probs, exploration=True, info=info)

        # We still need the distribution for log_prob
        # Since CategoricalHead already did softmax, probs are probabilities
        distribution = torch.distributions.Categorical(probs=probs)

        # Ensure action is the right shape for log_prob
        if action.dim() == 0:
            action_for_log_prob = action.unsqueeze(0)
        else:
            action_for_log_prob = action

        log_prob = distribution.log_prob(action_for_log_prob)

        return (
            action.squeeze(),
            log_prob.squeeze().item(),
            value.squeeze().item(),
        )
