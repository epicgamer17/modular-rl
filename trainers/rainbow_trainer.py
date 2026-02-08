"""
RainbowTrainer orchestrates the training process for Rainbow DQN by coordinating
the executor for data collection, the replay buffer for storage, and the learner
for optimization.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np

from executors import LocalExecutor, TorchMPExecutor
from agents.learners.rainbow_learner import RainbowLearner
from agents.policies.direct_policy import DirectPolicy
from agents.action_selectors.selectors import EpsilonGreedy
from agents.dqn_actor import DQNActor
from replay_buffers.transition import Transition, TransitionBatch
from modules.agent_nets.rainbow_dqn import RainbowNetwork
from stats.stats import StatTracker, PlotType
from utils.utils import update_linear_schedule, update_inverse_sqrt_schedule


class RainbowTrainer:
    """
    RainbowTrainer orchestrates the training process by coordinating
    the executor for data collection, the replay buffer for storage,
    and the learner for optimization.
    """

    def __init__(
        self,
        config,
        env,
        device: torch.device,
        stats: Optional[StatTracker] = None,
        test_agents: Optional[List] = None,
    ):
        """
        Initializes the RainbowTrainer.

        Args:
            config: RainbowConfig with hyperparameters.
            env: Environment instance or factory function.
            device: Torch device for training.
            stats: Optional StatTracker for logging metrics.
            test_agents: Optional list of agents to test against.
        """
        self.config = config
        self.device = device
        self.stats = (
            stats if stats is not None else StatTracker(model_name=config.model_name)
        )
        self.test_agents = test_agents if test_agents is not None else []

        self._env = env

        # Detect player_id for PettingZoo environments
        if hasattr(env, "possible_agents") and len(env.possible_agents) > 0:
            self._player_id = env.possible_agents[0]
        elif hasattr(env, "agents") and len(env.agents) > 0:
            self._player_id = env.agents[0]
        else:
            self._player_id = "player_0"

        # Get observation/action specs
        obs_dim, obs_dtype = self._determine_observation_dimensions(env)
        obs_dtype = getattr(config, "observation_dtype", obs_dtype)
        num_actions = self._get_num_actions(env)

        # 1. Initialize Networks
        input_shape = torch.Size((config.minibatch_size,) + obs_dim)
        self.model = RainbowNetwork(
            config=config,
            output_size=num_actions,
            input_shape=input_shape,
        )
        self.target_model = RainbowNetwork(
            config=config,
            output_size=num_actions,
            input_shape=input_shape,
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
            num_actions=num_actions,
            observation_dimensions=obs_dim,
            observation_dtype=obs_dtype,
        )
        self.buffer = self.learner.replay_buffer

        # 5. Initialize Executor
        if config.multi_process:
            self.executor = TorchMPExecutor()
        else:
            self.executor = LocalExecutor()

        # Launch workers (default to 1 worker if not specified)
        num_workers = getattr(config, "num_workers", 1)
        worker_args = (
            config.game.make_env,
            self.policy,
            config.game.num_players,
        )
        self.executor.launch(DQNActor, worker_args, num_workers)

        self.training_step = 0
        self.model_name = config.model_name

        # Intervals
        total_steps = config.training_steps
        self.checkpoint_interval = max(total_steps // 30, 1)
        self.test_interval = max(total_steps // 30, 1)
        self.test_trials = 5

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

    def _run_tests(self) -> None:
        """
        Runs test episodes and logs results.
        """
        dir = Path("checkpoints", self.model_name)
        step_dir = Path(dir, f"step_{self.training_step}")

        test_results = self.test(self.test_trials, dir=step_dir)
        if test_results:
            self.stats.append("test_score", test_results.get("score"))
            print(f"Test score: {test_results.get('score'):.3f}")

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
        """
        Saves model weights and stats.
        """
        import gc
        import os
        import dill as pickle

        base_dir = Path("checkpoints", self.model_name)
        step_dir = base_dir / f"step_{self.training_step}"
        os.makedirs(step_dir, exist_ok=True)

        # Save weights
        weights_dir = step_dir / "model_weights"
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = weights_dir / "weights.pt"
        checkpoint = {
            "training_step": self.training_step,
            "model_name": self.model_name,
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.learner.optimizer.state_dict(),
            "epsilon": self.current_epsilon,
        }
        torch.save(checkpoint, weights_path)

        # Save Config
        config_dir = base_dir / "configs"
        os.makedirs(config_dir, exist_ok=True)
        if hasattr(self.config, "dump"):
            self.config.dump(f"{config_dir}/config.yaml")

        # Save Stats
        stats_dir = step_dir / "graphs_stats"
        os.makedirs(stats_dir, exist_ok=True)

        if hasattr(self, "stats"):
            with open(stats_dir / "stats.pkl", "wb") as f:
                pickle.dump(self.stats.get_data(), f)

            # Plot graphs
            graph_dir = base_dir / "graphs"
            os.makedirs(graph_dir, exist_ok=True)
            self.stats.plot_graphs(dir=graph_dir)

        gc.collect()
        abs_path = os.path.abspath(step_dir)
        print(f"Saved checkpoint at step {self.training_step} to {abs_path}")

    def _determine_observation_dimensions(self, env):
        """
        Infers input dimensions for the neural network.
        Ported from BaseAgent.determine_observation_dimensions.
        """
        import gymnasium as gym

        obs_space = env.observation_space

        if isinstance(obs_space, gym.spaces.Box):
            return torch.Size(obs_space.shape), obs_space.dtype
        elif isinstance(obs_space, gym.spaces.Discrete):
            return torch.Size((1,)), np.int32
        elif isinstance(obs_space, gym.spaces.Tuple):
            return torch.Size((len(obs_space.spaces),)), np.int32
        elif callable(obs_space):
            # For PettingZoo-style callable observation spaces
            player_id = getattr(self, "_player_id", "player_0")
            try:
                space = obs_space(player_id)
            except KeyError:
                if hasattr(env, "possible_agents") and env.possible_agents:
                    space = obs_space(env.possible_agents[0])
                else:
                    raise
            return torch.Size(space.shape), space.dtype
        else:
            return torch.Size(obs_space.shape), obs_space.dtype

    def _get_num_actions(self, env) -> int:
        """
        Determines action space properties.
        Ported from BaseAgent._setup_action_space.
        """
        import gymnasium as gym

        if isinstance(env.action_space, gym.spaces.Discrete):
            return int(env.action_space.n)
        elif callable(env.action_space):  # PettingZoo
            player_id = getattr(self, "_player_id", "player_0")
            return int(env.action_space(player_id).n)
        elif hasattr(self.config.game, "num_actions"):
            return self.config.game.num_actions
        else:
            # Box/Continuous
            return int(env.action_space.shape[0])

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
