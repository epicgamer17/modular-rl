from __future__ import annotations

import time
from typing import List, Optional

import torch

from agents.action_selectors.factory import SelectorFactory
from agents.learner.base import UniversalLearner
from agents.learner.batch_iterators import RepeatSampleIterator
from agents.learner.callbacks import ResetNoiseCallback
from agents.trainers.base_trainer import BaseTrainer
from agents.workers.actors import get_actor_class
from losses.losses import ImitationLoss, LossPipeline
from modules.agent_nets.modular import ModularAgentNetwork
from modules.utils import get_lr_scheduler
from replay_buffers.buffer_factories import create_nfsp_buffer
from stats.stats import PlotType, StatTracker


class ImitationTrainer(BaseTrainer):
    """Supervised policy imitation trainer using UniversalLearner.

    Notes:
    - This trainer expects a config that is compatible with BaseTrainer (i.e. has `game`,
      execution params, and an `action_selector` block) and whose network can be built by
      `ModularAgentNetwork` (e.g. a config with `agent_type='supervised'`).
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

        # Network
        self.agent_network = ModularAgentNetwork(
            config=config,
            num_actions=self.num_actions,
            input_shape=self.obs_dim,
        ).to(device)
        if getattr(config, "kernel_initializer", None) is not None:
            self.agent_network.initialize(config.kernel_initializer)

        if getattr(config, "multi_process", False):
            self.agent_network.share_memory()

        # Action selector (used by actors to generate demonstrations)
        self.action_selector = SelectorFactory.create(
            config.action_selector.config_dict
        )

        # Buffer: (observations, legal_moves_mask, target_policies)
        self.buffer = create_nfsp_buffer(
            observation_dimensions=self.obs_dim,
            max_size=config.replay_buffer_size,
            num_actions=self.num_actions,
            batch_size=config.minibatch_size,
            observation_dtype=self.obs_dtype,
        )

        # Optimizer / LR schedule
        from torch.optim.adam import Adam
        from torch.optim.sgd import SGD

        if config.optimizer == Adam:
            optimizer = config.optimizer(
                params=self.agent_network.parameters(),
                lr=config.learning_rate,
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == SGD:
            optimizer = config.optimizer(
                params=self.agent_network.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")

        lr_scheduler = get_lr_scheduler(optimizer, config)

        # Loss
        loss_pipeline = LossPipeline([ImitationLoss(config, device, self.num_actions)])
        loss_pipeline.validate_dependencies(
            network_output_keys={"policies"},
            target_keys={"target_policies"},
        )

        self.learner = UniversalLearner(
            config=config,
            agent_network=self.agent_network,
            device=device,
            num_actions=self.num_actions,
            observation_dimensions=self.obs_dim,
            observation_dtype=self.obs_dtype,
            target_builder=None,
            loss_pipeline=loss_pipeline,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            clipnorm=config.clipnorm,
            callbacks=[ResetNoiseCallback()],
        )
        self.learner.replay_buffer = self.buffer

        # Executor + actors
        from agents.executors.factory import create_executor

        self.executor = create_executor(config)
        self.actor_cls = get_actor_class(env, config)
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
        self.executor.launch(self.actor_cls, worker_args, config.num_workers)

    def train_step(self) -> None:
        # 1) Broadcast weights
        self.executor.update_weights(self.agent_network.state_dict())

        # 2) Collect data (actors push directly to the buffer)
        _, collect_stats = self.executor.collect_data(
            min_samples=None, worker_type=self.actor_cls
        )
        for key, val in collect_stats.items():
            self.stats.append(key, val)

        # 3) Learner updates
        if self.buffer.size >= self.config.min_replay_buffer_size:
            for _ in range(self.config.num_minibatches):
                iterator = RepeatSampleIterator(
                    self.buffer, self.config.training_iterations, self.device
                )
                for step_metrics in self.learner.step(batch_iterator=iterator):
                    self._record_learner_metrics(step_metrics)

        self.training_step += 1

    def train(self) -> None:
        self._setup_stats()
        print(f"Starting Imitation training for {self.config.training_steps} steps...")
        last_log_time = time.time()

        while self.training_step < self.config.training_steps:
            self.train_step()

            if self.training_step % 10 == 0:
                elapsed = time.time() - last_log_time
                last_log_time = time.time()
                print(
                    f"Step {self.training_step}/{self.config.training_steps}, "
                    f"Time/10 steps: {elapsed:.2f}s"
                )
                self.stats.drain_queue()

            if self.training_step % self.checkpoint_interval == 0:
                self._save_checkpoint()

            if self.training_step % self.test_interval == 0:
                self.trigger_test(self.agent_network.state_dict(), self.training_step)

        self.stop_test()
        self.executor.stop()
        self._save_checkpoint()
        print("Training finished.")

    def _save_checkpoint(self) -> None:
        super()._save_checkpoint({})

    def _setup_stats(self) -> None:
        super()._setup_stats()
        for key in ["loss", "learner_fps", "actor_fps"]:
            if key not in self.stats.stats:
                self.stats._init_key(key)
        self.stats.add_plot_types("loss", PlotType.ROLLING_AVG, rolling_window=100)
