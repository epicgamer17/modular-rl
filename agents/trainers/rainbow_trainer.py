import torch
import time
from typing import Optional, List, Dict, Any, Tuple

from agents.trainers.base_trainer import BaseTrainer
from agents.learner.base import UniversalLearner

from agents.factories.action_selector import SelectorFactory
from modules.models.agent_network import AgentNetwork
from replay_buffers.transition import TransitionBatch, Transition
from stats.stats import StatTracker, PlotType
from utils.schedule import create_schedule
from agents.learner.target_builders import (
    TemporalDifferenceBuilder,
)
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from agents.learner.losses import QBootstrappingLoss
from agents.factories.replay_buffer import create_dqn_buffer
from agents.learner.batch_iterators import RepeatSampleIterator
from agents.factories.learner import build_universal_learner
from agents.action_selectors.policy_sources import NetworkPolicySource


class RainbowTrainer(BaseTrainer):
    """
    RainbowTrainer orchestrates the training process for Rainbow DQN.
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
        super().__init__(config, env, device, name, stats, test_agents)

        # 1. Initialize Networks
        from agents.factories.builders import make_backbone_fn, make_head_fn

        representation_fn = make_backbone_fn(getattr(config, "representation_backbone", None))
        head_fns = {
            name: make_head_fn(h_cfg)
            for name, h_cfg in config.heads.items()
        }

        network_kwargs = {
            "input_shape": self.obs_dim,
            "num_actions": self.num_actions,
            "representation_fn": representation_fn,
            "head_fns": head_fns,
            "num_players": getattr(config.game, "num_players", 1),
        }
        self.agent_network = AgentNetwork(**network_kwargs)
        self.target_agent_network = AgentNetwork(**network_kwargs)

        # Initialize weights
        if config.kernel_initializer is not None:
            self.agent_network.initialize(config.kernel_initializer)

        self.agent_network.to(device)

        if config.optimizer == Adam:
            self.optimizer = config.optimizer(
                params=self.agent_network.parameters(),
                lr=config.learning_rate,
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == SGD:
            self.optimizer = config.optimizer(
                params=self.agent_network.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")

        # Initialize target network
        from modules.utils import update_target_network

        update_target_network(self.agent_network, self.target_agent_network)
        self.agent_network.train()
        self.target_agent_network.eval()

        if config.multi_process:
            self.agent_network.share_memory()

        # 2. Initialize Action Selector
        self.action_selector = SelectorFactory.create(
            config.action_selector.config_dict
        )

        if hasattr(config, "epsilon_schedule") and config.epsilon_schedule is not None:
            self.epsilon_schedule = create_schedule(config.epsilon_schedule)
        else:
            self.epsilon_schedule = None

        # 3. Create support for distributional RL (C51)
        # Note: RainbowNetwork.initial_inference now handles calculating expected value from support
        # So we don't need to pass support explicitly to the selector in the old way
        # 9. Callbacks
        from agents.learner.callbacks import (
            TargetNetworkSyncCallback,
            ResetNoiseCallback,
            PriorityUpdaterCallback,
            EpsilonGreedySchedulerCallback,
        )
        from modules.utils import get_lr_scheduler

        # 3. Initialize LR Scheduler
        self.lr_scheduler = get_lr_scheduler(self.optimizer, config)

        # 4. Initialize Action Selector for loss calculation (greedy for Double DQN)
        from agents.action_selectors.selectors import ArgmaxSelector

        self.training_selector = ArgmaxSelector()

        self.buffer = create_dqn_buffer(
            observation_dimensions=self.obs_dim,
            max_size=config.replay_buffer_size,
            num_actions=self.num_actions,
            batch_size=config.minibatch_size,
            observation_dtype=self.obs_dtype,
            config=config,
        )

        self.schedules: Dict[str, Any] = {}
        per_beta_schedule_config = getattr(config, "per_beta_schedule", None)
        self.schedules["per_beta"] = create_schedule(per_beta_schedule_config)
        self.schedules["epsilon"] = create_schedule(config.epsilon_schedule)
        # 4. Initialize Learner (using factory)
        self.learner = build_universal_learner(
            config=config,
            agent_network=self.agent_network,
            device=device,
            target_agent_network=self.target_agent_network,
            priority_update_fn=self.buffer.update_priorities,
            set_beta_fn=self.buffer.set_beta,
            per_beta_schedule=self.schedules["per_beta"],
            epsilon_schedule=self.schedules["epsilon"],
            validator_params={
                "minibatch_size": config.minibatch_size,
                "unroll_steps": getattr(config, "unroll_steps", 0),
                "num_actions": self.num_actions,
                "num_players": getattr(config.game, "num_players", 1),
                "atom_size": config.atom_size if hasattr(config, "atom_size") else 1,
            },
        )

        # 5. Initialize Executor
        from agents.factories.executor import create_executor

        self.executor = create_executor(config)

        # 6. Initialize Workers
        from agents.workers.actors import RolloutActor

        self.policy_source = NetworkPolicySource(self.agent_network)

        # Decisions between single and vector adapter
        adapter_cls = self._get_adapter_class()
        env_factory = config.game.env_factory
        adapter_args = (env_factory,)
        worker_args = (
            adapter_cls,
            adapter_args,
            self.agent_network,
            self.policy_source,
            self.buffer,
            self.action_selector,
            getattr(config, "actor_device", "cpu"),
            getattr(config.game, "num_actions", None),
            getattr(config.game, "num_players", 1),
        )

        num_workers = config.num_workers if hasattr(config, "num_workers") else 1
        self.executor.launch_workers(RolloutActor, worker_args, num_workers=num_workers)

        # 6. Compile networks for the learner (main process)
        if config.compilation.enabled:
            if device.type == "mps":
                print("Skipping torch.compile on Apple Silicon (MPS).")
            else:
                self.agent_network = torch.compile(
                    self.agent_network,
                    mode=config.compilation.mode,
                    fullgraph=config.compilation.fullgraph,
                )
                # Optionally compile target network
                self.target_agent_network = torch.compile(
                    self.target_agent_network,
                    mode=config.compilation.mode,
                    fullgraph=config.compilation.fullgraph,
                )

    @property
    def current_epsilon(self) -> float:
        if "epsilon" in self.schedules:
            return self.schedules["epsilon"].get_value()
        return 0.0

    def train(self) -> None:
        """
        Main training loop.
        """
        self._setup_stats()

        print(f"Starting Rainbow training for {self.config.training_steps} steps...")
        start_time = time.time()

        while self.training_step < self.config.training_steps:
            self.train_step()

            # Poll for test results
            self.poll_test()

            # Periodic logging
            if self.training_step % 100 == 0 and self.training_step > 0:
                print(
                    f"Step {self.training_step}, "
                    f"Epsilon: {self.current_epsilon:.4f}, "
                    f"Buffer: {self.buffer.size}"
                )

        self.stop_test()
        self._save_checkpoint()
        print("Training finished.")

    def train_step(self) -> None:
        """Single training step for Rainbow: update epsilon, collect, and optimize."""
        # 1. Update weights and epsilon before collection
        self.executor.update_parameters(
            weights=self.agent_network.state_dict(),
            hyperparams={"epsilon": self.current_epsilon},
        )

        # 2. Collect data via actor
        from agents.workers.actors import RolloutActor

        # We collected 'replay_interval' steps per training iteration
        # If asynchronous, we might want to collect as much as possible
        results, collection_stats = self.executor.collect_data(
            num_steps=getattr(self.config, "replay_interval", 4),
            worker_type=RolloutActor,
        )

        # 3. Log collection stats
        for res in results:
            for score in res.get("batch_scores", []):
                self.stats.append("score", float(score))
            for length in res.get("batch_lengths", []):
                self.stats.append("episode_length", float(length))

        # 5. Learning step
        if self.buffer.size >= self.config.min_replay_buffer_size:
            for _ in range(self.config.num_minibatches):
                iterator = RepeatSampleIterator(
                    self.buffer, self.config.training_iterations, self.device
                )
                for step_metrics in self.learner.step(iterator):
                    self._record_learner_metrics(step_metrics)

            self.training_step += 1

            # 8. Periodic checkpointing
            if self.training_step % self.checkpoint_interval == 0:
                self._save_checkpoint()

            # 9. Periodic testing
            if self.training_step % self.test_interval == 0:
                self.trigger_test(self.agent_network.state_dict(), self.training_step)

    def _save_checkpoint(self) -> None:
        """Saves Rainbow checkpoint."""
        super()._save_checkpoint({})

    def _setup_stats(self) -> None:
        """
        Initializes the stat tracker with all required keys and plot types.
        """
        super()._setup_stats()
        stat_keys = [
            "loss",
            "learner_fps",
            "actor_fps",
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
        self.stats.add_plot_types("loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types(
            "episode_length", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "learner_fps", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types("actor_fps", PlotType.ROLLING_AVG, rolling_window=100)
