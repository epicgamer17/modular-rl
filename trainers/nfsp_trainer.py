"""
NFSPTrainer orchestrates the training process for NFSP by coordinating
the executor for data collection, the learner for optimization, and
handling the training loop.
"""

import time
import gc
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import random

import torch
import numpy as np
import dill as pickle

from executors import LocalExecutor, TorchMPExecutor
from agents.learners.nfsp_learner import NFSPLearner
from agents.policies.nfsp_policy import NFSPPolicy
from agents.action_selectors.selectors import EpsilonGreedy, CategoricalSelector
from agents.actors import GenericActor
from modules.agent_nets.rainbow_dqn import RainbowNetwork
from modules.agent_nets.policy_imitation import SupervisedNetwork
from stats.stats import StatTracker, PlotType


class NFSPTrainer:
    """
    NFSPTrainer orchestrates the NFSP training process.
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
        Initializes the NFSPTrainer.

        Args:
            config: NFSPConfig with hyperparameters.
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

        # 1. Initialize Networks
        # RL Network (Best Response)
        rl_config = config.rl_configs[0]
        self.br_model = RainbowNetwork(
            config=rl_config,
            input_shape=torch.Size((rl_config.minibatch_size,) + obs_dim),
            output_size=num_actions,
        ).to(device)

        self.br_target_model = RainbowNetwork(
            config=rl_config,
            input_shape=torch.Size((rl_config.minibatch_size,) + obs_dim),
            output_size=num_actions,
        ).to(device)
        self.br_target_model.load_state_dict(self.br_model.state_dict())

        # SL Network (Average Strategy)
        sl_config = config.sl_configs[0]
        self.avg_model = SupervisedNetwork(
            config=sl_config,
            output_size=num_actions,
            input_shape=torch.Size((sl_config.minibatch_size,) + obs_dim),
        ).to(device)

        if getattr(config, "multi_process", False):
            self.br_model.share_memory()
            self.br_target_model.share_memory()
            self.avg_model.share_memory()

        # 2. Initialize Action Selectors
        self.br_selector = EpsilonGreedy(epsilon=rl_config.eg_epsilon)
        self.avg_selector = CategoricalSelector(from_logits=False)

        # 3. Initialize Policy
        self.policy = NFSPPolicy(
            best_response_model=self.br_model,
            average_model=self.avg_model,
            best_response_selector=self.br_selector,
            average_selector=self.avg_selector,
            device=device,
            eta=config.anticipatory_param,
        )

        # 4. Initialize Learner
        self.learner = NFSPLearner(
            config=config,
            best_response_model=self.br_model,
            best_response_target_model=self.br_target_model,
            average_model=self.avg_model,
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
        worker_args = (
            self.config.game.make_env,
            self.policy,
            config.game.num_players,
        )
        self.executor.launch(GenericActor, worker_args, num_workers)

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

        print(f"Starting NFSP training for {self.config.training_steps} steps...")

        last_log_time = time.time()

        while self.training_step < self.config.training_steps:
            # 1. Update worker weights and parameters (eta)
            self.executor.update_weights(
                {
                    "best_response_state_dict": self.br_model.state_dict(),
                    "average_state_dict": self.avg_model.state_dict(),
                    "eta": self.config.anticipatory_param,  # Could be decayed here
                }
            )

            # 2. Collect data from workers
            # In NFSP, we collect games, then process their histories
            games, collection_stats = self.executor.collect_data(
                self.config.replay_interval
            )

            # Log collection stats
            for key, val in collection_stats.items():
                self.stats.append(key, val)

            for game in games:
                # Process game history to store transitions in learner's buffers
                if not game.policy_history:
                    continue
                policy_used = game.policy_history[
                    0
                ]  # Policy is fixed per episode in NFSPPolicy.reset

                for i in range(len(game.action_history)):
                    obs = game.observation_history[i]
                    info = game.info_history[i]
                    action = game.action_history[i]
                    reward = game.rewards[i]
                    next_obs = game.observation_history[i + 1]
                    next_info = game.info_history[i + 1]
                    done = i == len(game.action_history) - 1

                    self.learner.store(
                        observation=obs,
                        info=info,
                        action=action,
                        reward=reward,
                        next_observation=next_obs,
                        next_info=next_info,
                        done=done,
                        policy_used=policy_used,
                    )

            # 3. Learning step
            for _ in range(self.config.num_minibatches):
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
                    f"Step {self.training_step}/{self.config.training_steps}, "
                    f"Avg Score: {avg_score:.2f}, "
                    f"Time/10 steps: {elapsed:.2f}s"
                )
                last_log_time = time.time()

                # Ensure stats are processed if using multiprocessing
                self.stats.drain_queue()

            # 4. Update Target Networks (periodically)
            if self.training_step % self.config.rl_configs[0].transfer_interval == 0:
                self.learner.update_target_network()

            # 5. Periodic checkpointing
            if self.training_step % self.checkpoint_interval == 0:
                self._save_checkpoint()

            # 6. Periodic testing
            if self.training_step % self.test_interval == 0:
                self._run_tests()

        self.executor.stop()
        self._save_checkpoint()
        print("Training finished.")

    def _run_tests(self) -> None:
        """Runs evaluation episodes."""
        dir = Path("checkpoints", self.model_name)
        step_dir = Path(dir, f"step_{self.training_step}")
        test_results = self.test(self.test_trials, dir=step_dir)
        if test_results:
            self.stats.append("test_score", test_results.get("score"))
            print(f"Test score: {test_results.get('score'):.3f}")

    def test(self, num_trials: int, dir: str = "./checkpoints") -> Dict[str, float]:
        """Runs evaluation episodes."""
        if num_trials == 0:
            return {}

        test_env = self.config.game.make_env()
        scores = []

        # For testing, we usually use the Average Strategy (fully mixed)
        original_eta = self.policy.eta
        self.policy.eta = 0.0  # Force average strategy

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
                        # Track reward for the agent we're testing (usually player 0)
                        # In AEC, rewards are per-agent. test_env.rewards[player_id]
                        episode_reward += float(test_env.rewards[self._player_id])
                    else:
                        state, reward, terminated, truncated, info = test_env.step(
                            action_val
                        )
                        episode_reward += float(reward)

                    done = terminated or truncated

                scores.append(episode_reward)

        test_env.close()
        self.policy.eta = original_eta  # Restore

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
            "br_model": self.br_model.state_dict(),
            "avg_model": self.avg_model.state_dict(),
            "rl_optimizer": self.learner.rl_learner.optimizer.state_dict(),
            "sl_optimizer": self.learner.sl_optimizer.state_dict(),
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
        stat_keys = ["score", "rl_loss", "sl_loss", "test_score"]
        for key in stat_keys:
            if key not in self.stats.stats:
                self.stats._init_key(key)
        self.stats.add_plot_types("score", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("test_score", PlotType.BEST_FIT_LINE)
