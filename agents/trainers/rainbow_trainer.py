import torch
import time
from typing import Optional, List, Dict, Any, Tuple

from agents.trainers.base_trainer import BaseTrainer
from agents.learners.rainbow_learner import RainbowLearner
from agents.policies.direct_policy import DirectPolicy
from agents.action_selectors.selectors import EpsilonGreedy
from agents.actors.actors import get_actor_class
from modules.agent_nets.rainbow_dqn import RainbowNetwork
from replay_buffers.transition import TransitionBatch, Transition
from stats.stats import StatTracker, PlotType
from utils.utils import update_linear_schedule, update_inverse_sqrt_schedule


class RainbowTrainer(BaseTrainer):
    """
    RainbowTrainer orchestrates the training process for Rainbow DQN.
    """

    def __init__(
        self,
        config,
        env,
        device: torch.device,
        stats: Optional[StatTracker] = None,
        test_agents: Optional[List] = None,
    ):
        super().__init__(config, env, device, stats, test_agents)

        # 1. Initialize Networks
        self.model = RainbowNetwork(
            config=config,
            output_size=self.num_actions,
            input_shape=self.obs_dim,
        )
        self.target_model = RainbowNetwork(
            config=config,
            output_size=self.num_actions,
            input_shape=self.obs_dim,
        )

        # Initialize weights
        if config.kernel_initializer is not None:
            self.model.initialize(config.kernel_initializer)

        self.model.to(device)
        self.target_model.to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        if config.multi_process:
            self.model.share_memory()

        # 2. Initialize Action Selector with initial epsilon
        self.action_selector = EpsilonGreedy(epsilon=config.eg_epsilon)
        self.current_epsilon = config.eg_epsilon

        # 3. Create support for distributional RL (C51)
        self.support = None
        if config.atom_size > 1:
            self.support = torch.linspace(
                config.v_min, config.v_max, config.atom_size, device=device
            )

        # 4. Initialize Policy
        self.policy = DirectPolicy(
            model=self.model,
            action_selector=self.action_selector,
            device=device,
            support=self.support,
        )

        # 4. Initialize Learner
        self.learner = RainbowLearner(
            config=config,
            model=self.model,
            target_model=self.target_model,
            device=device,
            num_actions=self.num_actions,
            observation_dimensions=self.obs_dim,
            observation_dtype=self.obs_dtype,
        )
        self.buffer = self.learner.replay_buffer

        # 5. Initialize Executor
        from agents.executors.local_executor import LocalExecutor
        from agents.executors.torch_mp_executor import TorchMPExecutor

        if getattr(config, "multi_process", False):
            self.executor = TorchMPExecutor()
        else:
            self.executor = LocalExecutor()

        # Launch workers (default to 1 worker if not specified)
        num_workers = getattr(config, "num_workers", 1)
        worker_args = (
            config.game.make_env,
            self.policy,
            config.game.num_players,
            config,
        )
        actor_cls = get_actor_class(env)
        self.executor.launch(actor_cls, worker_args, num_workers)

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
                self.model.state_dict(),
                params={"epsilon": self.current_epsilon},
            )

            # 3. Collect data from executor (returns TransitionBatch objects)
            data, collect_stats = self.executor.collect_data(min_samples=1)

            # 4. Store transitions in buffer
            for batch in data:
                self._store_transitions(batch)

            # 5. Log collection stats
            for key, val in collect_stats.items():
                self.stats.append(key, val)

            # 6. Learning step
            if self.buffer.size >= self.config.min_replay_buffer_size:
                for _ in range(self.config.num_minibatches):
                    loss_stats = self.learner.step(self.stats)
                    if loss_stats:
                        for key, val in loss_stats.items():
                            self.stats.append(key, val)

                self.training_step += 1

                # 7. Update target network
                if self.training_step % self.config.transfer_interval == 0:
                    self.learner.update_target_network()

                # 8. Periodic checkpointing
                if self.training_step % self.checkpoint_interval == 0:
                    self._save_checkpoint()

                # 9. Periodic testing
                if self.training_step % self.test_interval == 0:
                    self._run_tests()

            # Periodic logging
            if self.training_step % 100 == 0 and self.training_step > 0:
                print(
                    f"Step {self.training_step}, "
                    f"Epsilon: {self.current_epsilon:.4f}, "
                    f"Buffer: {self.buffer.size}"
                )

        self.executor.stop()
        self._save_checkpoint()
        print("Training finished.")

    def _update_epsilon(self) -> None:
        """
        Updates epsilon according to the configured decay schedule.
        """
        if self.config.eg_epsilon_decay_type == "linear":
            self.current_epsilon = update_linear_schedule(
                self.config.eg_epsilon_final,
                self.config.eg_epsilon_final_step,
                self.config.eg_epsilon,
                self.training_step,
            )
        elif self.config.eg_epsilon_decay_type == "inverse_sqrt":
            self.current_epsilon = update_inverse_sqrt_schedule(
                self.config.eg_epsilon,
                self.training_step,
            )
        else:
            raise ValueError(
                f"Invalid epsilon decay type: {self.config.eg_epsilon_decay_type}"
            )

    def _store_transitions(self, batch: TransitionBatch) -> None:
        """
        Stores transitions in the replay buffer.

        Args:
            batch: TransitionBatch containing individual transitions.
        """
        for transition in batch:
            self._store_transition(transition)

    def _store_transition(self, transition: Transition) -> None:
        """
        Stores a single transition in the replay buffer.

        Args:
            transition: Single Transition object.
        """
        self.buffer.store(
            observations=transition.observation,
            actions=transition.action,
            rewards=transition.reward,
            next_observations=transition.next_observation,
            next_infos=transition.next_info if transition.next_info else {},
            dones=transition.done,
        )

    def test(self, num_trials: int, dir: str = "./checkpoints") -> Dict[str, float]:
        """
        Runs evaluation episodes and returns test scores.

        Args:
            num_trials: Number of evaluation episodes.
            dir: Directory for saving results.

        Returns:
            Dictionary with 'score', 'max_score', 'min_score' keys.
        """
        if num_trials == 0:
            return {}

        test_env = self.config.game.make_env()
        scores = []

        with torch.inference_mode():
            for _ in range(num_trials):
                state, info = test_env.reset()
                done = False
                episode_reward = 0.0
                episode_length = 0

                while not done and episode_length < 1000:
                    episode_length += 1

                    # Use policy with exploration disabled
                    action = self.policy.compute_action(state, info)
                    action_val = action.item() if hasattr(action, "item") else action

                    state, reward, terminated, truncated, info = test_env.step(
                        action_val
                    )
                    episode_reward += reward
                    done = terminated or truncated

                scores.append(episode_reward)

        test_env.close()

        if not scores:
            return {}

        return {
            "score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
        }

    def _save_checkpoint(self) -> None:
        """Saves Rainbow checkpoint."""
        checkpoint_data = {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.learner.optimizer.state_dict(),
            "epsilon": self.current_epsilon,
        }
        super()._save_checkpoint(checkpoint_data)

    def load_checkpoint_weights(self, checkpoint: Dict[str, Any]):
        """Loads Rainbow weights and epsilon."""
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        if "target_model" in checkpoint:
            self.target_model.load_state_dict(checkpoint["target_model"])
        if "optimizer" in checkpoint:
            self.learner.optimizer.load_state_dict(checkpoint["optimizer"])
        if "epsilon" in checkpoint:
            self.current_epsilon = checkpoint["epsilon"]

    def select_test_action(self, state, info, env) -> Any:
        """Greedy action for testing."""
        return self.policy.compute_action(state, info).item()

    def _setup_stats(self) -> None:
        """
        Initializes the stat tracker with all required keys and plot types.
        """
        stat_keys = [
            "score",
            "loss",
            "test_score",
            "episode_length",
            "learner_fps",
            "actor_fps",
        ]

        for key in stat_keys:
            if key not in self.stats.stats:
                if "test_score" in key:
                    self.stats._init_key(key, subkeys=["avg", "min", "max"])
                else:
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
