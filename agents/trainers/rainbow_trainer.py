import torch
import time
from typing import Optional, List, Dict, Any, Tuple

from agents.trainers.base_trainer import BaseTrainer
from agents.learners.rainbow_learner import RainbowLearner

from agents.action_selectors.factory import SelectorFactory
from agents.actors.actors import get_actor_class
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
        from agents.executors.local_executor import LocalExecutor
        from agents.executors.torch_mp_executor import TorchMPExecutor

        if config.multi_process:
            self.executor = TorchMPExecutor()
        else:
            self.executor = LocalExecutor()

        num_workers = config.num_workers
        worker_args = (
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
        Updates epsilon according to the configured schedule.
        """
        self.epsilon_schedule.step()
        self.current_epsilon = self.epsilon_schedule.get_value()
        self.action_selector.update_parameters({"epsilon": self.current_epsilon})

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
            next_legal_moves=(
                transition.next_legal_moves if transition.next_legal_moves else []
            ),
            terminated=transition.terminated,
            truncated=transition.truncated,
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

                    # Use action selector with exploration disabled/greedy?
                    # Test usually implies greedy. EpsilonGreedySelector with epsilon=0?
                    # Or just use the selector as is but update epsilon to 0 temporarily?
                    # Or use ArgmaxSelector?
                    # The standard way is to assume the selector is configured for test or we pass kwargs?
                    # BaseActionSelector.select_action takes **kwargs.
                    # EpsilonGreedySelector doesn't explicitly take 'epsilon' in select_action overrides yet?
                    # Let's assume we can update it or passing kwargs works if selector supports it.
                    # My EpsilonGreedySelector implementation uses self.epsilon.
                    # I should probably update it to support override.
                    # For now, I'll rely on update_parameters or a temporary selector for test?
                    # Efficient way: Just use argmax on value if network provides it.
                    # But proper way is using selector.
                    # If I use self.action_selector, it has self.epsilon.
                    # Use a separate test selector?
                    # Or just manually select greedy here since we know it's evaluation?
                    # "select_test_action" usually does greedy.
                    # Let's use the selector but we need to force greedy.
                    # If it's EpsilonGreedy, we can't easily force it without changing state.
                    # I'll manually do argmax here for safety as Rainbow test is greedy.

                    # Direct replacement of self.policy.compute_action(state, info)
                    # We want GREEDY action.
                    # network_output = self.agent_network.initial_inference(obs)
                    # action = network_output.q_values.argmax()

                    # BETTER: Use the selector mechanism but update params?
                    # Doing manual argmax mimics DirectPolicy behavior for test.
                    net_out = self.agent_network.obs_inference(state)
                    action = net_out.q_values.argmax(dim=-1)

                    action_val = action.item()

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

    def select_test_action(self, state, info, env) -> Any:
        """Greedy action for testing."""
        # Manual greedy selection using model
        net_out = self.agent_network.obs_inference(state)
        action = net_out.q_values.argmax(dim=-1)
        return action.item()

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
