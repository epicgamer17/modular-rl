import torch
import time
from typing import Optional, List, Dict, Any
from agents.trainers.base_trainer import BaseTrainer
from agents.learners.muzero_learner import MuZeroLearner
from agents.action_selectors.selectors import CategoricalSelector
from agents.action_selectors.decorators import MCTSDecorator
from search.search_factories import create_mcts
from agents.actors.actors import get_actor_class
from modules.agent_nets.muzero import MuZeroNetwork
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
        model_name: str = "agent",
        stats: Optional[StatTracker] = None,
        test_agents: List = None,
    ):
        super().__init__(config, env, device, model_name, stats, test_agents)
        # 1. Initialize Network
        # ... (network initialization)
        # The local import `from modules.agent_nets.muzero import AgentNetwork as Network` is removed
        # as MuZeroNetwork is already imported at the top and will be used directly.

        self.agent_network = MuZeroNetwork(
            config,
            input_shape=self.obs_dim,
            num_actions=self.num_actions,
        ).to(device)

        if config.kernel_initializer is not None:
            self.agent_network.initialize(config.kernel_initializer)

        if getattr(config, "multi_process", False):
            self.agent_network.share_memory()

        # 2. Initialize Search Algorithm (MCTS)
        from search.search_factories import create_mcts

        self.search_alg = create_mcts(config, device, self.num_actions)

        # 3. Initialize Action Selector (MCTS)
        # Inner: Chooses action from MCTS distribution (Categorical)
        inner_selector = CategoricalSelector()
        # Decorator: Runs MCTS and applies temperature
        self.action_selector = MCTSDecorator(
            inner_selector=inner_selector,
            search_algorithm=self.search_alg,
            config=config,
        )

        # Policy object removed. Learner needs a reference?
        # MuZeroLearner previously took 'policy'. Let's check if it needs it.
        # It used policy.preprocess/predict?
        # If Learner uses policy, we might need to update Learner or provide a shim.
        # Assuming for now we pass None or remove it if Learner allows.
        # Checking MuZeroLearner signature... (viewed earlier).
        # It takes 'policy' arg.
        # We'll check MuZeroLearner next. For now, passing None.

        # 4. Initialize Learner
        self.learner = MuZeroLearner(
            config=config,
            agent_network=self.agent_network,
            device=device,
            num_actions=self.num_actions,
            observation_dimensions=self.obs_dim,
            observation_dtype=self.obs_dtype,
            # policy=self.policy, # REMOVED/Updated
        )

        self.buffer = self.learner.replay_buffer

        # 6. Initialize Executor
        from agents.executors.local_executor import LocalExecutor
        from agents.executors.torch_mp_executor import TorchMPExecutor

        if config.multi_process:
            self.executor = TorchMPExecutor()
        else:
            self.executor = LocalExecutor()

        # Launch workers
        num_workers = config.num_workers
        worker_args = (
            config.game.make_env,
            self.agent_network,
            self.action_selector,
            config.game.num_players,
            config,
            device,
            self.model_name,
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
                    self.executor.update_weights(self.agent_network.state_dict())

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
            "agent_network": self.agent_network.state_dict(),
            "optimizer": self.learner.optimizer.state_dict(),
            "scheduler": self.learner.lr_scheduler.state_dict(),
        }
        super()._save_checkpoint(checkpoint_data)

    def load_checkpoint_weights(self, checkpoint: Dict[str, Any]):
        """Loads MuZero weights."""
        if "agent_network" in checkpoint:
            self.agent_network.load_state_dict(checkpoint["agent_network"])
        if "optimizer" in checkpoint:
            self.learner.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.learner.lr_scheduler.load_state_dict(checkpoint["scheduler"])

    def select_test_action(self, state, info, env) -> Any:
        """Search and select greedy action for testing."""
        # Use predict which handles Gumbel/Sequential Halving correctly

    def select_test_action(self, state, info, env) -> Any:
        """Search and select greedy action for testing."""
        # For testing, we might want temperature=0 or argmax.
        # MCTSDecorator handles temperature via config.
        # If we want explicit greedy selection:
        # We can pass exploration=False to select_action if Inner is Categorical?
        # CategoricalSelector(exploration=True) by default samples.
        # We might want to force argmax for test.
        # Or pass a specific low temperature (handled by MCTS Decorator logic usually).
        # For now, relying on select_action with episode_step=0 (or high) might not be enough override.
        # We'll just call select_action.

        # Note: MCTS Decorator expects 'episode_step' in kwargs for temperature.
        # During test, maybe we want 'temperature' to be 0 or small.
        # But MCTS stochasticity is often desired? Usually Argmax on visit counts.
        # If we use ArgmaxSelector as inner for Test?
        # But self.action_selector is fixed.
        # We may need a way to override behavior.

        # Passing 'exploration=False' to CategoricalSelector forces argmax.
        # MCTSDecorator passes kwargs to inner.
        action, _ = self.action_selector.select_action(
            self.agent_network, state, info, exploration=False, episode_step=1e9
        )
        return action.item() if hasattr(action, "item") else action

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
