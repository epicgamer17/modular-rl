import torch
import time
from typing import Optional, List, Dict, Any
from agents.trainers.base_trainer import BaseTrainer
from agents.learners.muzero_learner import MuZeroLearner
from agents.policies.search_policy import SearchPolicy
from agents.action_selectors.selectors import TemperatureSelector
from search.search_factories import create_mcts
from agents.actors.actors import get_actor_class
from modules.agent_nets.muzero import AgentNetwork
from stats.stats import StatTracker, PlotType


class MuZeroTrainer(BaseTrainer):
    """
    MuZeroTrainer orchestrates the training process for MuZero.
    """

    def __init__(
        self,
        config,
        env,
        device: torch.device,
        stats: Optional[StatTracker] = None,
        test_agents: List = None,
    ):
        super().__init__(config, env, device, stats, test_agents)
        # 1. Initialize Network
        # ... (network initialization)
        from modules.agent_nets.muzero import AgentNetwork as Network

        self.model = Network(
            config,
            input_shape=self.obs_dim,
            num_actions=self.num_actions,
        ).to(device)

        if config.kernel_initializer is not None:
            self.model.initialize(config.kernel_initializer)

        if getattr(config, "multi_process", False):
            self.model.share_memory()

        # 2. Initialize Search Algorithm (MCTS)
        from search.search_factories import create_mcts

        self.search_alg = create_mcts(config, device, self.num_actions)

        # 3. Initialize Policy
        self.policy = SearchPolicy(
            model=self.model,
            search_algorithm=self.search_alg,
            config=config,
            device=device,
            observation_dimensions=self.obs_dim,
        )

        # 4. Initialize Learner
        self.learner = MuZeroLearner(
            config=config,
            model=self.model,
            device=device,
            num_actions=self.num_actions,
            observation_dimensions=self.obs_dim,
            observation_dtype=self.obs_dtype,
            policy=self.policy,
        )

        self.buffer = self.learner.replay_buffer

        self.buffer = self.learner.replay_buffer

        # 6. Initialize Executor
        from agents.executors.local_executor import LocalExecutor
        from agents.executors.torch_mp_executor import TorchMPExecutor

        if getattr(config, "multi_process", False):
            self.executor = TorchMPExecutor()
        else:
            self.executor = LocalExecutor()

        # Launch workers
        num_workers = getattr(config, "num_workers", 1)
        worker_args = (
            config.game.make_env,
            self.policy,
            config.game.num_players,
            config,
        )
        from agents.actors.actors import get_actor_class

        actor_cls = get_actor_class(env)
        self.executor.launch(actor_cls, worker_args, num_workers)

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
            for sequence in data:
                self.buffer.store_aggregate(sequence_object=sequence)

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

            # Drain stats queue to avoid deadlock if workers are logging
            if hasattr(self.stats, "drain_queue"):
                self.stats.drain_queue()

        self.executor.stop()
        # Final checkpoint and stats plot
        self._save_checkpoint()
        print("Training finished.")

    def _save_checkpoint(self) -> None:
        """Saves MuZero checkpoint."""
        checkpoint_data = {
            "model": self.model.state_dict(),
            "optimizer": self.learner.optimizer.state_dict(),
            "scheduler": self.learner.lr_scheduler.state_dict(),
        }
        super()._save_checkpoint(checkpoint_data)

    def load_checkpoint_weights(self, checkpoint: Dict[str, Any]):
        """Loads MuZero weights."""
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.learner.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.learner.lr_scheduler.load_state_dict(checkpoint["scheduler"])

    def select_test_action(self, state, info, env) -> Any:
        """Search and select greedy action for testing."""
        # Use predict which handles Gumbel/Sequential Halving correctly
        best_action, info_dict = self.policy.predict(state, info, env=env)
        return best_action

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
            "chance_probs",
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
        self.stats.add_plot_types("chance_probs", PlotType.BAR)
