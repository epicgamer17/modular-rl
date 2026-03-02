import torch
import time
from typing import Optional, List, Dict, Any, Tuple

from agents.trainers.base_trainer import BaseTrainer
from agents.learners.rainbow_learner import RainbowLearner

from agents.action_selectors.factory import SelectorFactory
from agents.workers.actors import get_actor_class
from modules.agent_nets.modular import ModularAgentNetwork
from replay_buffers.transition import TransitionBatch, Transition
from stats.stats import StatTracker, PlotType
from utils.schedule import create_schedule


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
        self.agent_network = ModularAgentNetwork(
            config=config,
            num_actions=self.num_actions,
            input_shape=self.obs_dim,
        )
        self.target_agent_network = ModularAgentNetwork(
            config=config,
            num_actions=self.num_actions,
            input_shape=self.obs_dim,
        )

        # Initialize weights
        if config.kernel_initializer is not None:
            self.agent_network.initialize(config.kernel_initializer)

        self.agent_network.to(device)
        self.target_agent_network.to(device)
        self.target_agent_network.load_state_dict(self.agent_network.state_dict())
        self.target_agent_network.eval()

        if config.multi_process:
            self.agent_network.share_memory()

        # 2. Initialize Action Selector
        self.action_selector = SelectorFactory.create(
            config.action_selector.config_dict
        )

        # Initialize epsilon schedule
        self.epsilon_schedule = create_schedule(config.epsilon_schedule)
        self.current_epsilon = self.epsilon_schedule.get_value()

        # 3. Create support for distributional RL (C51)
        # Note: RainbowNetwork.initial_inference now handles calculating expected value from support
        # So we don't need to pass support explicitly to the selector in the old way

        # 4. Initialize Learner
        self.learner = RainbowLearner(
            config=config,
            agent_network=self.agent_network,
            target_agent_network=self.target_agent_network,
            device=device,
            num_actions=self.num_actions,
            observation_dimensions=self.obs_dim,
            observation_dtype=self.obs_dtype,
        )
        self.buffer = self.learner.replay_buffer

        # 5. Initialize Executor
        from agents.executors.factory import create_executor

        self.executor = create_executor(config)

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

        # 6. Compile networks for the learner (main process)
        if config.compilation.enabled:
            self.agent_network.compile(
                mode=config.compilation.mode, fullgraph=config.compilation.fullgraph
            )
            # Optionally compile target network
            self.target_agent_network.compile(
                mode=config.compilation.mode, fullgraph=config.compilation.fullgraph
            )

    def train(self) -> None:
        """
        Main training loop.
        """
        self._setup_stats()

        print(f"Starting Rainbow training for {self.config.training_steps} steps...")
        start_time = time.time()

        while self.training_step < self.config.training_steps:
            # 1. Update epsilon schedule
            self._update_epsilon()

            # 2. Broadcast weights and epsilon to workers
            self.executor.update_weights(
                self.agent_network.state_dict(),
                params={"epsilon": self.current_epsilon},
            )

            # 3. Wait for data to be collected
            # The actors push directly to the buffer.
            # We use collect_data just to retrieve their stats and sync.
            _, collect_stats = self.executor.collect_data(
                min_samples=None, worker_type=self.actor_cls
            )

            # 4. Log collection stats
            for key, val in collect_stats.items():
                self.stats.append(key, val)

            # 5. Learning step
            if self.buffer.size >= self.config.min_replay_buffer_size:
                for _ in range(self.config.num_minibatches):
                    loss_stats = self.learner.step(self.stats)
                    if loss_stats:
                        for key, val in loss_stats.items():
                            self.stats.append(key, val)

                self.training_step += 1

                # 6. Update target network
                if self.training_step % self.config.transfer_interval == 0:
                    self.learner.update_target_network()

                # 8. Periodic checkpointing
                if self.training_step % self.checkpoint_interval == 0:
                    self._save_checkpoint()

                # 9. Periodic testing
                if self.training_step % self.test_interval == 0:
                    self.trigger_test(
                        self.agent_network.state_dict(), self.training_step
                    )

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
        self.executor.stop()
        self._save_checkpoint()
        print("Training finished.")

    def _update_epsilon(self) -> None:
        """
        Updates epsilon according to the configured schedule.
        """
        self.epsilon_schedule.step()
        self.current_epsilon = self.epsilon_schedule.get_value()
        self.action_selector.update_parameters({"epsilon": self.current_epsilon})

    def _save_checkpoint(self) -> None:
        """Saves Rainbow checkpoint."""
        checkpoint_data = {
            "agent_network": self.agent_network.state_dict(),
            "target_agent_network": self.target_agent_network.state_dict(),
            "optimizer": self.learner.optimizer.state_dict(),
            "epsilon": self.current_epsilon,
        }
        super()._save_checkpoint(checkpoint_data)

    def load_checkpoint_weights(self, checkpoint: Dict[str, Any]):
        """Loads Rainbow weights and epsilon."""
        if "agent_network" in checkpoint:
            self.agent_network.load_state_dict(checkpoint["agent_network"])
        if "target_agent_network" in checkpoint:
            self.target_agent_network.load_state_dict(
                checkpoint["target_agent_network"]
            )
        if "optimizer" in checkpoint:
            self.learner.optimizer.load_state_dict(checkpoint["optimizer"])
        if "epsilon" in checkpoint:
            self.current_epsilon = checkpoint["epsilon"]

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
