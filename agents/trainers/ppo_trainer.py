import torch
import time
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
from agents.trainers.base_trainer import BaseTrainer
from agents.executors.local_executor import LocalExecutor
from agents.executors.torch_mp_executor import TorchMPExecutor
from agents.factories.action_selector import SelectorFactory
from agents.action_selectors.policy_sources import NetworkPolicySource



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
        from agents.factories.builders import make_backbone_fn, make_head_fn

        # New standard: input_shape excludes batch dimension
        input_shape = self.obs_dim

        # Build functional components
        representation_fn = make_backbone_fn(getattr(config, "representation_backbone", None))
        memory_core_fn = make_backbone_fn(getattr(config, "prediction_backbone", None))

        head_fns = {}
        if hasattr(config, "heads") and config.heads:
            for name, head_cfg in config.heads.items():
                head_fns[name] = make_head_fn(head_cfg)

        self.agent_network = AgentNetwork(
            input_shape=input_shape,
            num_actions=self.num_actions,
            representation_fn=representation_fn,
            memory_core_fn=memory_core_fn,
            head_fns=head_fns,
            num_players=getattr(config.game, "num_players", 1),
        )
        self.agent_network.to(device)

        # Initialize weights
        if getattr(config, "kernel_initializer", None) is not None:
            self.agent_network.initialize(config.kernel_initializer)

        if config.multi_process:
            self.agent_network.share_memory()

        # 2. Initialize Action Selector and Policy Source
        from agents.action_selectors.decorators import PPODecorator
        raw_selector = SelectorFactory.create(
            config.action_selector.config_dict
        )
        self.action_selector = PPODecorator(raw_selector)
        self.policy_source = NetworkPolicySource(self.agent_network)

        # 3. Initialize Replay Buffer
        from agents.factories.replay_buffer import create_ppo_buffer

        self.replay_buffer = create_ppo_buffer(
            observation_dimensions=self.obs_dim,
            max_size=config.replay_buffer_size,
            gamma=config.discount_factor,
            gae_lambda=config.gae_lambda,
            num_actions=self.num_actions,
            observation_dtype=self.obs_dtype,
        )

        # 4. Initialize Executor
        from agents.factories.executor import create_executor

        self.executor = create_executor(config)

        # 5. Initialize Learner via Factory
        from agents.factories.learner import build_universal_learner

        self.learner = build_universal_learner(
            config=config,
            agent_network=self.agent_network,
            device=device,
            weight_broadcast_fn=self.executor.update_parameters,
            validator_params={
                "minibatch_size": config.minibatch_size,
                "num_actions": self.num_actions,
                "num_players": getattr(config.game, "num_players", 1),
            },
        )

        # 6. Initialize Workers
        from agents.workers.actors import RolloutActor

        # Prepare worker args
        # For PPO, we use the GymAdapter for the environment
        adapter_cls = self._get_adapter_class()
        env_factory = config.game.env_factory

        worker_args = (
            adapter_cls,
            (env_factory,),
            self.agent_network,
            self.policy_source,
            self.replay_buffer,
            self.action_selector,
            getattr(config, "actor_device", "cpu"),
            getattr(config.game, "num_actions", None),
            getattr(config.game, "num_players", 1),
        )
        
        # Launch rollout workers
        num_workers = config.num_workers if hasattr(config, "num_workers") else 1
        self.executor.launch_workers(RolloutActor, worker_args, num_workers=num_workers)

        # 7. Compile network for the learner (main process)
        if config.compilation.enabled:
            if device.type == "mps":
                print("Skipping torch.compile on Apple Silicon (MPS).")
            else:
                self.agent_network = torch.compile(
                    self.agent_network,
                    mode=config.compilation.mode,
                    fullgraph=config.compilation.fullgraph,
                )

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
        # 1. Update weights before collection
        self.executor.update_parameters(weights=self.agent_network.state_dict())

        # 2. Collect trajectory data via actor
        from agents.workers.actors import RolloutActor
        
        # collect_data will request 'steps_per_epoch' from RolloutActor
        # Results is a list of dicts (one per worker)
        results, collection_stats = self.executor.collect_data(
            num_steps=self.config.steps_per_epoch,
            worker_type=RolloutActor
        )

        # 3. Log collection stats
        self._record_collection_metrics(results)


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
