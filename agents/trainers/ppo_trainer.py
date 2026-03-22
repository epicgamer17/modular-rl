import torch
import time
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
from agents.trainers.base_trainer import BaseTrainer
from agents.executors.local_executor import LocalExecutor
from agents.executors.torch_mp_executor import TorchMPExecutor
from agents.action_selectors.factory import SelectorFactory
from agents.action_selectors.policy_sources import NetworkPolicySource

# from agents.workers.actors import get_actor_class # REMOVED as unused

from modules.models.agent_network import AgentNetwork

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
        name: str = "agent",
        stats: Optional[StatTracker] = None,
        test_agents: Optional[List] = None,
    ):
        """
        Initializes the PPOTrainer.

        Args:
            config: PPOConfig with hyperparameters.
            env: Environment instance or factory function.
            device: Torch device for training.
            name: Name for this model (used for checkpoints, stats, videos).
            stats: Optional StatTracker for logging metrics.
            test_agents: Optional list of agents to test against.
        """
        super().__init__(config, env, device, name, stats, test_agents)

        # 1. Initialize Network
        # New standard: input_shape excludes batch dimension
        input_shape = self.obs_dim
        self.agent_network = AgentNetwork(
            input_shape=input_shape,
            num_actions=self.num_actions,
            arch_config=config.arch,
            representation_config=getattr(config, "representation_backbone", None),
            prediction_backbone_config=getattr(config, "prediction_backbone", None),
            heads_config=config.heads,
            num_players=getattr(config.game, "num_players", 1),
        )
        self.agent_network.to(device)

        # Initialize weights
        if getattr(config, "kernel_initializer", None) is not None:
            self.agent_network.initialize(config.kernel_initializer)

        if config.multi_process:
            self.agent_network.share_memory()

        # 2. Initialize Action Selector and Policy Source
        self.action_selector = SelectorFactory.create(
            config.action_selector.config_dict
        )
        self.policy_source = NetworkPolicySource(self.agent_network)

        # 3. Initialize Replay Buffer
        from replay_buffers.buffer_factories import create_ppo_buffer

        self.replay_buffer = create_ppo_buffer(
            observation_dimensions=self.obs_dim,
            max_size=config.replay_buffer_size,
            gamma=config.discount_factor,
            gae_lambda=config.gae_lambda,
            num_actions=self.num_actions,
            observation_dtype=self.obs_dtype,
        )

        # 4. Initialize Executor
        from agents.executors.factory import create_executor

        self.executor = create_executor(config)

        # 5. Initialize Learner via Factory
        from agents.learner.factory import build_universal_learner

        self.learner = build_universal_learner(
            config=config,
            agent_network=self.agent_network,
            device=device,
            weight_broadcast_fn=self.executor.update_weights,
            validator_params={
                "minibatch_size": config.minibatch_size,
                "num_actions": self.num_actions,
            },
        )

        # 6. Compile network for the learner (main process)
        if config.compilation.enabled:
            if device.type == "mps":
                print("Skipping torch.compile on Apple Silicon (MPS).")
            else:
                self.agent_network = torch.compile(
                    self.agent_network,
                    mode=config.compilation.mode,
                    fullgraph=config.compilation.fullgraph,
                )

        # Note: We do not launch actor workers here because PPOTrainer currently uses an
        # inline data collection loop within `train()`. Launching workers would cause them
        # to execute `play_sequence()`, which PPOLearner's buffer does not currently support
        # (raises NotImplementedError for sequence processing).

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
            self.train_step()

            # 4. Periodic checkpointing
            if self.training_step % self.checkpoint_interval == 0:
                self._save_checkpoint()

            # 5. Periodic testing
            if self.training_step % self.test_interval == 0:
                self.trigger_test(self.agent_network.state_dict(), self.training_step)

            # Poll for background test results
            self.poll_test()

            # Periodic logging
            if self.training_step % 10 == 0:
                # We'll need a way to get avg_score if we want to keep this print here.
                # For now, just print the step.
                print(f"Step {self.training_step}")

            # Drain stats queue to avoid deadlock if workers are logging
            if hasattr(self.stats, "drain_queue"):
                self.stats.drain_queue()

        self.stop_test()
        # stop_test() already calls executor.stop() and sets it to None
        self._save_checkpoint()
        print("Training finished.")

    def train_step(self) -> None:
        """
        Single training step for PPO: collects a full epoch of data and optimizes.
        """
        # 1. Collect trajectory data (steps_per_epoch transitions)
        steps_collected = 0
        trajectory_start_index = 0
        completed_scores = []
        completed_lengths = []

        while steps_collected < self.config.steps_per_epoch:
            with torch.no_grad():
                # Get state from environment
                state, info = self._env.reset()
                done = False
                current_episode_score = 0.0
                current_episode_length = 0

                while not done and steps_collected < self.config.steps_per_epoch:
                    obs_tensor = torch.tensor(
                        state, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    result = self.policy_source.get_inference(obs=obs_tensor, info=info)
                    action, metadata = self.action_selector.select_action(
                        result=result,
                        info=info,
                        exploration=True,
                    )

                    log_prob = metadata.get("log_prob")
                    value = metadata.get("value")

                    assert (
                        log_prob is not None
                    ), f"log_prob is None. Metadata: {metadata}"
                    assert value is not None, f"value is None. Metadata: {metadata}"

                    action_val = action.item()

                    # Environment step
                    next_state, reward, terminated, truncated, next_info = (
                        self._env.step(action_val)
                    )
                    done = terminated or truncated

                    # Store transition
                    self.replay_buffer.store(
                        observations=state,
                        actions=action_val,
                        values=float(value.item() if torch.is_tensor(value) else value),
                        old_log_probs=float(
                            log_prob.item() if torch.is_tensor(log_prob) else log_prob
                        ),
                        rewards=reward,
                        dones=done,
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
                if terminated:
                    last_value = 0.0
                else:
                    with torch.inference_mode():
                        obs = torch.tensor(state).unsqueeze(0).to(self.device)
                        out = self.agent_network.obs_inference(obs)
                        last_value = out.value.item()

                trajectory_end_index = self.replay_buffer.size
                trajectory_slice = slice(trajectory_start_index, trajectory_end_index)

                if trajectory_end_index > trajectory_start_index:
                    input_processor = self.replay_buffer.input_processor
                    result = input_processor.finish_trajectory(
                        self.replay_buffer.buffers,
                        trajectory_slice,
                        last_value=last_value,
                    )
                    if result:
                        for key, value in result.items():
                            self.replay_buffer.buffers[key][trajectory_slice] = value

                trajectory_start_index = trajectory_end_index

        # Log collection stats
        if completed_scores:
            for s, l in zip(completed_scores, completed_lengths):
                self.stats.append("score", float(s))
                self.stats.append("episode_length", float(l))

        # 3. Learning step
        from agents.learner.batch_iterators import PPOEpochIterator

        iterator = PPOEpochIterator(
            replay_buffer=self.replay_buffer,
            num_epochs=self.config.train_policy_iterations,
            num_minibatches=self.config.num_minibatches,
            device=self.device,
        )

        for step_metrics in self.learner.step(iterator):
            self._record_learner_metrics(step_metrics)

        # Clear buffer for next epoch
        self.replay_buffer.clear()

        self.training_step += 1

    def _save_checkpoint(self) -> None:
        """
        Saves model weights and stats using BaseTrainer implementation.
        """
        super()._save_checkpoint({})

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
