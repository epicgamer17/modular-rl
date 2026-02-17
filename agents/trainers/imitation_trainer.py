"""
ImitationTrainer orchestrates supervised policy imitation training by coordinating
the executor for data collection and the ImitationLearner for optimization.

This trainer uses DirectPolicy with CategoricalSelector or ArgmaxSelector
for action selection during data collection.
"""

import gc
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np
import dill as pickle

from agents.executors.local_executor import LocalExecutor
from agents.executors.torch_mp_executor import TorchMPExecutor
from agents.learners.imitation_learner import ImitationLearner
from agents.policies.direct_policy import DirectPolicy
from agents.action_selectors.selectors import CategoricalSelector, ArgmaxSelector
from agents.actors.actors import get_actor_class
from modules.agent_nets.policy_imitation import SupervisedNetwork
from stats.stats import StatTracker, PlotType


class ImitationTrainer:
    """
    ImitationTrainer orchestrates supervised policy imitation training.
    It collects expert/target policy data and trains a network to imitate it.
    """

    def __init__(
        self,
        config,
        env,
        device: torch.device,
        stats: Optional[StatTracker] = None,
        test_agents: Optional[List] = None,
        use_categorical: bool = True,
    ):
        """
        Initializes the ImitationTrainer.

        Args:
            config: Configuration with hyperparameters.
            env: Environment instance.
            device: Torch device for training.
            stats: Optional StatTracker for logging metrics.
            test_agents: Optional list of agents to test against.
            use_categorical: If True, use CategoricalSelector (sampling).
                             If False, use ArgmaxSelector (greedy).
        """
        self.config = config
        self.device = device
        self.stats = (
            stats if stats is not None else StatTracker(model_name=config.model_name)
        )
        self.test_agents = test_agents if test_agents is not None else []

        self._env = env
        assert not callable(
            env
        ), "env must be an environment instance, not a factory function"

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
        self.num_actions = num_actions

        # 1. Initialize Network
        # input_shape = torch.Size((config.minibatch_size,) + obs_dim)
        # New standard: input_shape excludes batch dimension
        input_shape = obs_dim
        self.model = SupervisedNetwork(
            config=config,
            output_size=num_actions,
            input_shape=input_shape,
        ).to(device)

        if getattr(config, "multi_process", False):
            self.model.share_memory()

        # 2. Initialize Action Selector
        if use_categorical:
            # SupervisedNetwork outputs softmax probabilities (not logits)
            self.action_selector = CategoricalSelector(from_logits=False)
        else:
            self.action_selector = ArgmaxSelector()

        # 3. Initialize Policy
        self.policy = DirectPolicy(
            model=self.model,
            action_selector=self.action_selector,
            device=device,
        )

        # 4. Initialize Learner
        self.learner = ImitationLearner(
            config=config,
            model=self.model,
            device=device,
            num_actions=num_actions,
            observation_dimensions=obs_dim,
            observation_dtype=obs_dtype,
        )

        # 5. Initialize Executor
        if getattr(config, "multi_process", False):
            self.executor = TorchMPExecutor()
        else:
            self.executor = LocalExecutor()

        # Launch workers
        num_workers = getattr(config, "num_workers", 1)
        num_players = getattr(config.game, "num_players", 1)
        worker_args = (
            config.game.make_env,
            self.policy,
            num_players,
        )
        actor_cls = get_actor_class(env)
        self.executor.launch(actor_cls, worker_args, num_workers)

        self.training_step = 0
        self.model_name = config.model_name

        # Intervals
        total_steps = getattr(config, "training_steps", 1000)
        self.checkpoint_interval = max(total_steps // 30, 1)
        self.test_interval = max(total_steps // 30, 1)
        self.test_trials = 5

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
            self.executor.update_weights(self.model.state_dict())

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
        if not hasattr(sequence, "action_history") or not sequence.action_history:
            return

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

    def _run_tests(self) -> None:
        """Runs evaluation episodes."""
        dir = Path("checkpoints", self.model_name)
        step_dir = Path(dir, f"step_{self.training_step}")
        test_results = self.test(self.test_trials, dir=step_dir)
        if test_results:
            self.stats.append("test_score", test_results.get("score"))
            print(f"Test score: {test_results.get('score'):.3f}")

    def test(self, num_trials: int, dir: str = "./checkpoints") -> Dict[str, float]:
        """
        Runs evaluation episodes.

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
        num_players = getattr(self.config.game, "num_players", 1)

        with torch.inference_mode():
            for _ in range(num_trials):
                # 1. Reset
                if num_players != 1:
                    test_env.reset()
                    state, reward, terminated, truncated, info = test_env.last()
                else:
                    state, info = test_env.reset()

                self.policy.reset(state)
                done = False
                episode_reward = 0.0

                while not done:
                    action = self.policy.compute_action(state, info)
                    action_val = action.item() if torch.is_tensor(action) else action

                    # 2. Step
                    if num_players != 1:
                        test_env.step(action_val)
                        state, reward, terminated, truncated, info = test_env.last()
                        episode_reward += float(test_env.rewards[self._player_id])
                    else:
                        state, reward, terminated, truncated, info = test_env.step(
                            action_val
                        )
                        episode_reward += float(reward)

                    done = terminated or truncated

                scores.append(episode_reward)

        test_env.close()

        return {
            "score": sum(scores) / len(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
        }

    def _save_checkpoint(self) -> None:
        """Saves model weights and stats."""
        base_dir = Path("checkpoints", self.model_name)
        step_dir = base_dir / f"step_{self.training_step}"
        os.makedirs(step_dir, exist_ok=True)

        weights_dir = step_dir / "model_weights"
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = weights_dir / "weights.pt"

        checkpoint = {
            "training_step": self.training_step,
            "model_name": self.model_name,
            "model": self.model.state_dict(),
            "optimizer": self.learner.optimizer.state_dict(),
        }
        torch.save(checkpoint, weights_path)

        if hasattr(self, "stats"):
            stats_dir = step_dir / "graphs_stats"
            os.makedirs(stats_dir, exist_ok=True)
            with open(stats_dir / "stats.pkl", "wb") as f:
                pickle.dump(self.stats.get_data(), f)
            graph_dir = base_dir / "graphs"
            os.makedirs(graph_dir, exist_ok=True)
            self.stats.plot_graphs(dir=graph_dir)

        gc.collect()
        abs_path = os.path.abspath(step_dir)
        print(f"Saved checkpoint at step {self.training_step} to {abs_path}")

    def _determine_observation_dimensions(self, env):
        """
        Infers input dimensions for the neural network.

        Args:
            env: The environment to inspect.

        Returns:
            Tuple of (observation_shape, observation_dtype).
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

        Args:
            env: The environment to inspect.

        Returns:
            Number of discrete actions.
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
        """Initializes the stat tracker with required keys and plot types."""
        stat_keys = ["score", "loss", "test_score", "learner_fps"]

        for key in stat_keys:
            if key not in self.stats.stats:
                self.stats._init_key(key)

        self.stats.add_plot_types(
            "score", PlotType.ROLLING_AVG, PlotType.BEST_FIT_LINE, rolling_window=100
        )
        self.stats.add_plot_types("test_score", PlotType.BEST_FIT_LINE)
        self.stats.add_plot_types("loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types(
            "learner_fps", PlotType.ROLLING_AVG, rolling_window=100
        )
