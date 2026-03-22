import torch
import time
from typing import Optional, List, Dict, Any
from agents.trainers.base_trainer import BaseTrainer
from agents.learner.factory import build_universal_learner
from agents.learner.batch_iterators import SingleBatchIterator
from replay_buffers.buffer_factories import create_muzero_buffer
from agents.action_selectors.selectors import CategoricalSelector
from agents.action_selectors.decorators import TemperatureSelector
from agents.workers.actors import get_actor_class
from modules.models.agent_network import AgentNetwork
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
        name: str = "agent",
        stats: Optional[StatTracker] = None,
        test_agents: List = None,
    ):
        super().__init__(config, env, device, name, stats, test_agents)

        # Create player_id_mapping for multi-player games
        if hasattr(env, "possible_agents"):
            self.player_id_mapping = {
                agent_id: i for i, agent_id in enumerate(env.possible_agents)
            }
        else:
            self.player_id_mapping = {"player_0": 0}

        # 1. Initialize Network
        self.agent_network = AgentNetwork(
            input_shape=self.obs_dim,
            num_actions=self.num_actions,
            arch_config=config.arch,
            representation_config=getattr(config, "representation_backbone", None),
            world_model_config=config.world_model,
            heads_config=config.heads,
            projector_config=config.projector,
            stochastic=config.stochastic,
            num_players=config.game.num_players,
            num_chance_codes=config.num_chance,
            validator_params={
                "minibatch_size": config.minibatch_size,
                "unroll_steps": config.unroll_steps,
                "num_actions": self.num_actions,
                "atom_size": (config.support_range * 2) + 1 if hasattr(config, "support_range") and config.support_range else 1,
            },
        ).to(device)

        if config.kernel_initializer is not None:
            self.agent_network.initialize(config.kernel_initializer)

        if config.multi_process:
            self.agent_network.share_memory()

        # 2. Initialize Action Selector (MCTS)
        inner_selector = CategoricalSelector()
        self.action_selector = TemperatureSelector(
            inner_selector=inner_selector,
            config=config,
        )

        # 3. The Facts (Replay Buffer)
        self.buffer = create_muzero_buffer(
            observation_dimensions=self.obs_dim,
            max_size=config.replay_buffer_size,
            num_actions=self.num_actions,
            num_players=config.game.num_players,
            player_id_mapping=self.player_id_mapping,
            unroll_steps=config.unroll_steps,
            n_step=config.n_step,
            gamma=config.discount_factor,
            batch_size=config.minibatch_size,
            observation_dtype=self.obs_dtype,
            alpha=config.per_alpha,
            beta=config.per_beta_schedule.initial,
            epsilon=config.per_epsilon,
            use_batch_weights=config.per_use_batch_weights,
            use_initial_max_priority=config.per_use_initial_max_priority,
            lstm_horizon_len=config.lstm_horizon_len,
            value_prefix=config.use_value_prefix,
            tau=config.reanalyze_tau,
            multi_process=config.multi_process,
            observation_quantization=config.observation_quantization,
            observation_compression=config.observation_compression,
        )
        # 5. Initialize Executor
        from agents.executors.factory import create_executor

        self.executor = create_executor(config)

        # 4. The Orchestrator (Universal Learner)
        self.learner = build_universal_learner(
            config=config,
            agent_network=self.agent_network,
            device=device,
            priority_update_fn=self.buffer.update_priorities,
            weight_broadcast_fn=self.executor.update_weights,
        )

        if config.multi_process:
            self.buffer.share_memory()

        # Launch workers
        num_workers = config.num_workers
        worker_args = (
            config.game.make_env,
            self.agent_network,
            self.action_selector,
            self.buffer,
            config.game.num_players,
            config,
            device,
            self.name,
        )
        self.actor_cls = get_actor_class(env, config)
        self.executor.launch(self.actor_cls, worker_args, num_workers)

        # 6. Compile network for the learner (main process)
        if config.compilation.enabled:
            self.agent_network.compile(
                mode=config.compilation.mode, fullgraph=config.compilation.fullgraph
            )

    def train(self):
        """Main training loop."""
        self._setup_stats()

        print(f"Starting training for {self.config.training_steps} steps...")
        start_time = time.time()

        while self.training_step < self.config.training_steps:
            self.train_step()

            # Poll for background test results
            self.poll_test()

            # Periodic logging
            if self.training_step % 100 == 0 and self.training_step > 0:
                print(f"Step {self.training_step}")

            # Drain stats queue to avoid deadlock
            if hasattr(self.stats, "drain_queue"):
                self.stats.drain_queue()

        self.stop_test()
        self._save_checkpoint()
        print("Training finished.")

    def train_step(self) -> Dict[str, Any]:
        """Perform one training step (batch of gradients)."""
        # 1. Wait for data to be collected
        _, collect_stats = self.executor.collect_data(
            min_samples=None, worker_type=self.actor_cls
        )

        # 2. Log collection stats
        for key, val in collect_stats.items():
            self.stats.append(key, val)

        # 3. Learning step
        if self.buffer.size >= self.config.min_replay_buffer_size:
            for _ in range(self.config.num_minibatches):
                iterator = SingleBatchIterator(self.buffer, self.device)
                for step_metrics in self.learner.step(iterator):
                    self._record_learner_metrics(step_metrics)

            self.training_step += 1

            # 4. Update workers (if needed)
            if self.training_step % self.config.transfer_interval == 0:
                self.executor.update_weights(self.agent_network.state_dict())

            # 5. Periodic checkpointing
            if self.training_step % self.checkpoint_interval == 0:
                self._save_checkpoint()

            # 6. Periodic testing
            if self.training_step % self.test_interval == 0:
                self.trigger_test(self.agent_network.state_dict(), self.training_step)

        return {}

    def _save_checkpoint(self) -> None:
        """Saves MuZero checkpoint."""
        super()._save_checkpoint({})

    def _setup_stats(self):
        """Initializes the stat tracker with all required keys and plot types."""
        from stats.stats import PlotType

        test_score_keys = (
            [f"vs_{agent.name}_score" for agent in self.test_agents]
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
            "mcts_sps",
            "chance_probs",
        ] + test_score_keys

        # Initialize keys
        for key in stat_keys:
            if key not in self.stats.stats:
                if key == "test_score":
                    self.stats._init_key(
                        key
                    )  # Removed min/max subkeys as they are no longer logged
                elif "_score" in key:
                    # Player-specific subkeys for vs_agent tests
                    num_players = self.config.game.num_players
                    subkeys = [f"p{i}" for i in range(num_players)] + ["avg"]
                    self.stats._init_key(key, subkeys=subkeys)
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
        self.stats.add_plot_types("mcts_sps", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("chance_probs", PlotType.BAR)
