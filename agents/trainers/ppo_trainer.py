import torch
import time
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
from agents.trainers.base_trainer import BaseTrainer
from agents.executors.local_executor import LocalExecutor
from agents.executors.torch_mp_executor import TorchMPExecutor
from agents.learners.ppo_learner import PPOLearner
from agents.learners.ppo_learner import PPOLearner
from agents.action_selectors.factory import SelectorFactory
from agents.actors.actors import get_actor_class
from modules.agent_nets.ppo import PPONetwork

# from agents.policies.ppo_policy import PPOPolicy # REMOVED
from stats.stats import StatTracker, PlotType


class PPOTrainer(BaseTrainer):
    """
    PPOTrainer orchestrates the training process by coordinating
    the executor for data collection and the learner for optimization.
    """

    def __init__(
        self,
        config,
        env,
        device: torch.device,
        model_name: str = "agent",
        stats: Optional[StatTracker] = None,
        test_agents: Optional[List] = None,
    ):
        """
        Initializes the PPOTrainer.

        Args:
            config: PPOConfig with hyperparameters.
            env: Environment instance or factory function.
            device: Torch device for training.
            model_name: Name for this model (used for checkpoints, stats, videos).
            stats: Optional StatTracker for logging metrics.
            test_agents: Optional list of agents to test against.
        """
        super().__init__(config, env, device, model_name, stats, test_agents)

        # 1. Initialize Network
        # New standard: input_shape excludes batch dimension
        input_shape = self.obs_dim
        self.model = PPONetwork(
            config=config,
            input_shape=input_shape,
        )
        self.model.to(device)

        # Initialize weights
        if config.kernel_initializer is not None:
            self.model.initialize(config.kernel_initializer)

        if config.multi_process:
            self.model.share_memory()

        # 2. Initialize Action Selector
        self.action_selector = SelectorFactory.create(
            config.action_selector.config_dict
        )

        # 3. Initialize Learner
        self.learner = PPOLearner(
            config=config,
            model=self.model,
            device=device,
            num_actions=self.num_actions,
            observation_dimensions=self.obs_dim,
            observation_dtype=self.obs_dtype,
        )

        # 5. Initialize Executor
        if config.multi_process:
            self.executor = TorchMPExecutor()
        else:
            self.executor = LocalExecutor()

        # Launch workers (default to 1 worker if not specified)
        num_workers = config.num_workers
        actor_cls = get_actor_class(env)
        worker_args = (
            config.game.make_env,
            self.model,
            self.action_selector,
            config.game.num_players,
            config,
            device,
            self.model_name,
        )
        self.executor.launch(actor_cls, worker_args, num_workers)

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
            trajectory_start_index = 0

            while steps_collected < self.config.steps_per_epoch:
                with torch.no_grad():
                    # Get state from environment
                    state, info = self._env.reset()
                    done = False
                    current_episode_score = 0.0
                    current_episode_length = 0

                    while not done and steps_collected < self.config.steps_per_epoch:
                        # Compute action and value
                        # action, log_prob, value = self.policy.compute_action_with_info(state, info)
                        obs_tensor = torch.tensor(
                            state, dtype=torch.float32, device=self.device
                        )
                        action, metadata = self.action_selector.select_action(
                            agent_network=self.model,
                            obs=obs_tensor,
                            info=info,
                            exploration=True,
                        )

                        log_prob = metadata.get("log_prob")
                        value = metadata.get("value")

                        action_val = (
                            action.item() if hasattr(action, "item") else action
                        )

                        # Environment step
                        next_state, reward, terminated, truncated, next_info = (
                            self._env.step(action_val)
                        )
                        done = terminated or truncated

                        # Store transition
                        self.learner.replay_buffer.store(
                            observations=state,
                            actions=action_val,
                            values=float(value.item() if torch.is_tensor(value) else value),
                            log_probabilities=float(
                                log_prob.item() if torch.is_tensor(log_prob) else log_prob
                            ),
                            rewards=reward,
                            info=info,
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
                            last_value, _ = self.model.value(obs)
                            last_value = last_value.item()

                    trajectory_end_index = self.learner.replay_buffer.size
                    trajectory_slice = slice(
                        trajectory_start_index, trajectory_end_index
                    )

                    if trajectory_end_index > trajectory_start_index:
                        input_processor = self.learner.replay_buffer.input_processor
                        result = input_processor.finish_trajectory(
                            self.learner.replay_buffer.buffers,
                            trajectory_slice,
                            last_value=last_value,
                        )
                        if result:
                            for key, value in result.items():
                                self.learner.replay_buffer.buffers[key][
                                    trajectory_slice
                                ] = value

                    trajectory_start_index = trajectory_end_index

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
            loss_stats = self.learner.step(self.stats, self.training_step)
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

            # Drain stats queue to avoid deadlock if workers are logging
            if hasattr(self.stats, "drain_queue"):
                self.stats.drain_queue()

        self.executor.stop()
        self._save_checkpoint()
        print("Training finished.")

    def _save_checkpoint(self) -> None:
        """
        Saves model weights and stats using BaseTrainer implementation.
        """
        checkpoint_data = {
            "model": self.model.state_dict(),
            "policy_optimizer": self.learner.policy_optimizer.state_dict(),
            "value_optimizer": self.learner.value_optimizer.state_dict(),
            "policy_scheduler": self.learner.policy_scheduler.state_dict(),
            "value_scheduler": self.learner.value_scheduler.state_dict(),
        }
        super()._save_checkpoint(checkpoint_data)

    def load_checkpoint_weights(self, checkpoint: Dict[str, Any]):
        """Ported from BaseAgent.load_model_weights and load_optimizer_state."""
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        if "policy_optimizer" in checkpoint:
            self.learner.policy_optimizer.load_state_dict(
                checkpoint["policy_optimizer"]
            )
        if "value_optimizer" in checkpoint:
            self.learner.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        if "policy_scheduler" in checkpoint:
            self.learner.policy_scheduler.load_state_dict(
                checkpoint["policy_scheduler"]
            )
        if "value_scheduler" in checkpoint:
            self.learner.value_scheduler.load_state_dict(checkpoint["value_scheduler"])

    def select_test_action(self, state, info, env) -> Any:
        # PPO usually tests with its policy (which might be greedy depending on compute_action implementation)
        # We enforce greedy/non-exploratory for test if possible, or sample if that's standard PPO (usually stochastic)
        # But for 'score' we usually want best effort.
        # CategoricalSelector has 'exploration' kwarg.
        obs_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        action, _ = self.action_selector.select_action(
            self.model, obs_tensor, info, exploration=False
        )
        return action.item() if hasattr(action, "item") else action

    def _setup_stats(self):
        """Initializes the stat tracker with PPO-specific keys and plot types."""
        super()._setup_stats()
        from stats.stats import PlotType

        stat_keys = [
            "policy_loss",
            "value_loss",
            "policy_entropy",
            "kl_divergence",
            "explained_variance",
            "learner_fps",
        ]

        # Initialize keys
        for key in stat_keys:
            if key not in self.stats.stats:
                self.stats._init_key(key)

        # Add plot types
        self.stats.add_plot_types(
            "policy_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "value_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "policy_entropy", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "kl_divergence", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "explained_variance", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "learner_fps", PlotType.ROLLING_AVG, rolling_window=100
        )
