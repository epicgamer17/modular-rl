import torch
from typing import Optional, List, Dict, Any, Tuple

from agents.trainers.base_trainer import BaseTrainer
from agents.learners.imitation_learner import ImitationLearner
from agents.action_selectors.factory import SelectorFactory
from agents.actors.actors import get_actor_class
from modules.agent_nets.policy_imitation import SupervisedNetwork
from stats.stats import StatTracker, PlotType
from agents.executors.local_executor import LocalExecutor
from agents.executors.torch_mp_executor import TorchMPExecutor
import torch
import numpy as np
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class ImitationTrainer(BaseTrainer):
    """
    ImitationTrainer orchestrates supervised policy imitation training.
    """

    def __init__(
        self,
        config,
        env,
        device: torch.device,
        name: str = "agent",
        stats: Optional[StatTracker] = None,
        test_agents: Optional[List] = None,
        use_categorical: bool = True,
    ):
        super().__init__(config, env, device, name, stats, test_agents)

        # 1. Initialize Network
        self.agent_network = SupervisedNetwork(
            config=config,
            output_size=self.num_actions,
            input_shape=self.obs_dim,
        ).to(device)

        if getattr(config, "multi_process", False):
            self.agent_network.share_memory()

        # 2. Initialize Action Selector
        self.action_selector = SelectorFactory.create(
            config.action_selector.config_dict
        )

        # 3. Initialize Policy - REMOVED
        # self.policy = DirectPolicy(...)

        # 4. Initialize Learner
        self.learner = ImitationLearner(
            config=config,
            agent_network=self.agent_network,
            device=device,
            num_actions=self.num_actions,
            observation_dimensions=self.obs_dim,
            observation_dtype=self.obs_dtype,
        )
        self.buffer = self.learner.replay_buffer

        # 5. Initialize Executor
        if config.multi_process:
            self.executor = TorchMPExecutor()
        else:
            self.executor = LocalExecutor()

        # Launch workers (default to 1 worker if not specified)
        num_workers = config.num_workers
        worker_args = (
            config.game.make_env,
            config.game.make_env,
            self.agent_network,
            self.action_selector,
            config.game.num_players,
            config,
            device,
            self.name,
        )
        actor_cls = get_actor_class(env)
        self.executor.launch(actor_cls, worker_args, num_workers)

    def train(self) -> None:
        """
        Main training loop.
        """
        self._setup_stats()

        training_steps = getattr(self.config, "training_steps", 1000)
        replay_interval = getattr(self.config, "replay_interval", 1)
        num_minibatches = getattr(self.config, "num_minibatches", 1)

        print(f"Starting Imitation training for {training_steps} steps...")
        last_log_time = time.time()

        while self.training_step < training_steps:
            # 1. Update worker weights
            self.executor.update_weights(self.agent_network.state_dict())

            # 2. Collect data from workers
            sequences, collection_stats = self.executor.collect_data(replay_interval)

            # Log collection stats
            for key, val in collection_stats.items():
                self.stats.append(key, val)

            # 3. Store transitions from collected sequences
            for sequence in sequences:
                self._store_sequence_transitions(sequence)

            # 4. Learning step
            for _ in range(num_minibatches):
                loss_stats = self.learner.step(self.stats)
                if loss_stats:
                    for key, val in loss_stats.items():
                        self.stats.append(key, val)

            self.training_step += 1

            # Periodic logging
            if self.training_step % 10 == 0:
                elapsed = time.time() - last_log_time
                avg_score = 0.0
                if "score" in self.stats.stats and self.stats.stats["score"]:
                    avg_score = np.mean(self.stats.stats["score"][-10:])

                print(
                    f"Step {self.training_step}/{training_steps}, "
                    f"Avg Score: {avg_score:.2f}, "
                    f"Time/10 steps: {elapsed:.2f}s"
                )
                last_log_time = time.time()
                self.stats.drain_queue()

            # 5. Periodic checkpointing
            if self.training_step % self.checkpoint_interval == 0:
                self._save_checkpoint()

            # 6. Periodic testing
            if self.training_step % self.test_interval == 0:
                self._run_tests()

        self.executor.stop()
        self._save_checkpoint()
        print("Training finished.")

    def _store_sequence_transitions(self, sequence) -> None:
        """
        Stores transitions from a sequence episode into the learner's buffer.

        For imitation learning, we store (observation, info, target_policy)
        where target_policy is typically the action taken as a one-hot vector.

        Args:
            sequence: Sequence object with observation_history, action_history, info_history.
        """
        for i in range(len(sequence.action_history)):
            obs = sequence.observation_history[i]
            info = sequence.info_history[i] if sequence.info_history else {}
            action = sequence.action_history[i]

            # Create one-hot target policy from action
            target_policy = torch.zeros(self.num_actions)
            target_policy[action] = 1.0

            self.learner.store(
                observation=obs,
                info=info,
                target_policy=target_policy,
            )

    def _save_checkpoint(self) -> None:
        """Saves Imitation checkpoint."""
        checkpoint_data = {
            "agent_network": self.agent_network.state_dict(),
            "optimizer": self.learner.optimizer.state_dict(),
        }
        super()._save_checkpoint(checkpoint_data)

    def load_checkpoint_weights(self, checkpoint: Dict[str, Any]):
        """Loads Imitation weights."""
        if "agent_network" in checkpoint:
            self.agent_network.load_state_dict(checkpoint["agent_network"])
        if "optimizer" in checkpoint:
            self.learner.optimizer.load_state_dict(checkpoint["optimizer"])

    def select_test_action(self, state, info, env) -> Any:
        """Select action for testing (from model)."""
        obs = torch.as_tensor(state, device=self.device, dtype=torch.float32).unsqueeze(
            0
        )
        with torch.inference_mode():
            # probs = self.agent_network(obs)
            inf_out = self.agent_network.obs_inference(obs)
            probs = inf_out.policy.probs
        action = probs.argmax(dim=-1).item()
        return action

    def _setup_stats(self) -> None:
        """Initializes the stat tracker with required keys and plot types."""
        stat_keys = ["score", "loss", "test_score", "learner_fps"]

        for key in stat_keys:
            if key not in self.stats.stats:
                if "test_score" in key:
                    self.stats._init_key(key, subkeys=["avg", "min", "max"])
                else:
                    self.stats._init_key(key)

        self.stats.add_plot_types(
            "score", PlotType.ROLLING_AVG, PlotType.BEST_FIT_LINE, rolling_window=100
        )
        self.stats.add_plot_types("test_score", PlotType.BEST_FIT_LINE)
        self.stats.add_plot_types("loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types(
            "learner_fps", PlotType.ROLLING_AVG, rolling_window=100
        )
