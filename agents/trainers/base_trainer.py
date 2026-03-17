import os
import gc
import torch
import numpy as np
import dill as pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
import gymnasium as gym
from stats.stats import StatTracker
from modules.utils import get_uncompiled_model


class BaseTrainer:
    """
    Base trainer class containing shared logic for all specialized trainers.
    Handles environment specification inference, stat setup, checkpointing, and testing.
    """

    def __init__(
        self,
        config: Any,
        env: Any,
        device: torch.device,
        name: str = "agent",
        stats: Optional[StatTracker] = None,
        test_agents: Optional[List] = None,
    ):
        self.config = config
        self.device = device
        self.name = name
        self.stats = stats if stats is not None else StatTracker(name=name)
        self.test_agents = test_agents if test_agents is not None else []
        self._env = env

        # Detect player_id for PettingZoo environments
        self._player_id = self._detect_player_id(env)

        # Get observation/action specs
        self.obs_dim, self.obs_dtype = self._determine_observation_dimensions(env)
        self.num_actions = self._get_num_actions(env)
        self.num_players = self._get_num_players(env)

        self.training_step = 0

        # Intervals
        total_steps = config.training_steps
        self.checkpoint_interval = max(total_steps // 30, 1)
        self.test_interval = max(total_steps // 30, 1)
        self.test_trials = 5

        # Executor state
        self.executor = None
        self._tester_launched = False
        self._tester_step = 0

    def setup(self):
        """Initializes stats and any other setup required before training."""
        self._setup_stats()

    def setup_tester(self):
        """Initializes the Tester using the TestFactory and launched via an Executor."""
        from agents.workers.tester import TestFactory, Tester
        from agents.executors.local_executor import LocalExecutor
        from agents.executors.torch_mp_executor import TorchMPExecutor

        # 1. Initialize Executor if not already done
        if self.executor is None:
            from agents.executors.factory import create_executor

            self.executor = create_executor(self.config)

        # 2. Prepare test types
        test_types = TestFactory.create_default_test_types(
            self.config, num_trials=self.test_trials
        )
        from agents.workers.tester import VsAgentTest

        for agent in self.test_agents:
            for player_idx in range(self.num_players):
                test_types.append(
                    VsAgentTest(
                        name=f"vs_{agent.name}_p{player_idx}",
                        num_trials=self.test_trials,
                        opponent=agent,
                        player_idx=player_idx,
                    )
                )

        # 3. Launch Tester
        # 1. Safely unwrap the network before passing across process boundaries
        uncompiled_network = get_uncompiled_model(self.agent_network)

        # 2. Launch Tester
        launch_args = TestFactory.get_launch_args(
            config=self.config,
            agent_network=uncompiled_network,  # Pass the uncompiled version!
            action_selector=self.action_selector,
            device=torch.device("cpu"),
            name=f"{self.name}_tester",
            test_types=test_types,
        )
        self.executor.launch(Tester, launch_args, num_workers=1)

    def trigger_test(self, state_dict: Dict[str, Any], step: int):
        """
        Triggers evaluation. For LocalExecutor, this runs immediately.
        For TorchMPExecutor, it ensures weights are synced.
        """
        from agents.workers.tester import Tester

        if not self._tester_launched:
            self.setup_tester()
            self._tester_launched = True
            if self.executor is None:
                return

        # Update step (weights are shared via shared_memory if multi_process=True)
        self._tester_step = step

        # Signal executor to run test (both local and multi-process need this now)
        self.executor.request_work(Tester)

        # If local, run synchronously now
        if not self.config.multi_process:
            # LocalExecutor._fetch_available_results runs play_sequence
            results, _ = self.executor.collect_data(min_samples=1, worker_type=Tester)
            for res in results:
                self._process_test_results(res, step)

    def poll_test(self):
        """Polls for background test results from the executor."""
        from agents.workers.tester import Tester

        if self.executor is None or not self.config.multi_process:
            return

        # Fetch whatever is available in the result queue for Tester
        results, _ = self.executor.collect_data(min_samples=None, worker_type=Tester)
        if results:
            # We only care about the most recent test result for logging
            self._process_test_results(results[-1], self._tester_step)

    def stop_test(self):
        """Stops the executor (stops everything)."""
        if self.executor is not None:
            self.executor.stop()
            self.executor = None

    def _process_test_results(self, all_results: Dict[str, Dict[str, Any]], step: int):
        """Logs results from all test types."""
        for test_name, res in all_results.items():
            # Standard evaluation (e.g., standard, self_play)
            if test_name == "self_play":
                # Only log p0_score for self_play to show consistent training progress
                if "p0_score" in res:
                    self.stats.append("test_score", res["p0_score"], subkey="p0")
                elif "score" in res:
                    self.stats.append("test_score", res["score"], subkey="avg")

            elif test_name == "standard":
                if "score" in res:
                    self.stats.append("test_score", res["score"], subkey="avg")

            # Vs Agent evaluations (e.g., vs_Random_p0, vs_Random_p1)
            elif test_name.startswith("vs_"):
                # Parse base agent name (e.g., vs_Random_p0 -> vs_Random)
                parts = test_name.split("_")
                # Expected format: vs_{agent_name}_p{idx}
                if (
                    len(parts) >= 3
                    and parts[-1].startswith("p")
                    and parts[-1][1:].isdigit()
                ):
                    agent_name = "_".join(parts[:-1])  # e.g. vs_Random
                    player_idx = parts[-1]  # e.g. p0

                    if "score" in res:
                        # Group by agent name, use player_idx as subkey
                        self.stats.append(
                            f"{agent_name}_score", res["score"], subkey=player_idx
                        )
                else:
                    # Fallback for non-indexed vs tests
                    if "score" in res:
                        self.stats.append(f"{test_name}_score", res["score"])

            print(f"[{test_name}] score: {res.get('score', 0):.3f} (step {step})")

    def _detect_player_id(self, env) -> str:
        if self.config.game.num_players > 1:
            return env.possible_agents[0]
        return "player_0"

    def _determine_observation_dimensions(self, env) -> Tuple[torch.Size, np.dtype]:
        """
        Infers input dimensions for the neural network.
        Ported from BaseAgent.determine_observation_dimensions.
        """
        obs_space = env.observation_space

        if isinstance(obs_space, gym.spaces.Box):
            return torch.Size(obs_space.shape), obs_space.dtype
        elif isinstance(obs_space, gym.spaces.Discrete):
            return torch.Size((1,)), np.int32
        elif isinstance(obs_space, gym.spaces.Tuple):
            return torch.Size((len(obs_space.spaces),)), np.int32
        elif callable(obs_space):
            space = obs_space(self._player_id)
            return torch.Size(space.shape), space.dtype
        else:
            return torch.Size(obs_space.shape), obs_space.dtype

    def _get_num_actions(self, env) -> int:
        """
        Determines action space properties.
        Ported from BaseAgent._setup_action_space.
        """
        if isinstance(env.action_space, gym.spaces.Discrete):
            return int(env.action_space.n)
        elif callable(env.action_space):
            return int(env.action_space(self._player_id).n)
        elif self.config.game.num_actions:
            return self.config.game.num_actions
        else:
            # Box/Continuous
            return int(env.action_space.shape[0])

    def _get_num_players(self, env) -> int:
        return self.config.game.num_players

    def _save_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """
        Saves model weights and stats. Standardized checkpoint format.
        """
        base_dir = Path("checkpoints", self.name)
        step_dir = base_dir / f"step_{self.training_step}"
        os.makedirs(step_dir, exist_ok=True)

        # Save learner state (weights, optimizer, step)
        weights_dir = step_dir / "model_weights"
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = weights_dir / "weights.pt"
        self.learner.save_checkpoint(str(weights_path))

        # Save Config
        config_dir = base_dir / "configs"
        os.makedirs(config_dir, exist_ok=True)
        self.config.dump(f"{config_dir}/config.yaml")

        # Save Stats
        stats_dir = step_dir / "graphs_stats"
        os.makedirs(stats_dir, exist_ok=True)

        with open(stats_dir / "stats.pkl", "wb") as f:
            pickle.dump(self.stats.get_data(), f)

        # Plot graphs
        graph_dir = base_dir / "graphs"
        os.makedirs(graph_dir, exist_ok=True)
        self.stats.plot_graphs(dir=graph_dir)

        gc.collect()
        abs_path = os.path.abspath(step_dir)
        print(f"Saved checkpoint at step {self.training_step} to {abs_path}")

    @classmethod
    def load_from_checkpoint(
        cls,
        env: Any,
        config_class: Type,
        dir_path: str,
        training_step: int,
        device: torch.device,
        **extra_init_kwargs,
    ):
        """
        Standardized loading logic for all trainers.
        """
        dir_path = Path(dir_path)
        step_dir = dir_path / f"step_{training_step}"
        weights_path = step_dir / "model_weights/weights.pt"
        config_path = dir_path / "configs/config.yaml"

        # 1. Load Config
        config = config_class.load(config_path)

        # 2. Instantiate Trainer
        trainer = cls(config=config, env=env, device=device, **extra_init_kwargs)
        trainer.training_step = training_step

        # 3. Load learner state from checkpoint
        trainer.learner.load_checkpoint(str(weights_path))

        # 4. Load Stats
        stats_path = step_dir / "graphs_stats/stats.pkl"
        if stats_path.exists():
            with open(stats_path, "rb") as f:
                trainer.stats.set_data(pickle.load(f))

        return trainer

    def load_checkpoint_weights(self, checkpoint: Dict[str, Any]):
        """Child classes should implement this to load specific weights from checkpoint dictionary."""
        raise NotImplementedError

    def _setup_stats(self):
        """Initializes the stat tracker with common keys and plot types."""
        from stats.stats import PlotType

        stat_keys = [
            "score",
            "episode_length",
            "test_score",
        ]

        # Add test_score vs other agents if applicable
        for agent in self.test_agents:
            stat_keys.append(f"vs_{agent.name}_score")

        # Initialize keys
        player_subkeys = [f"p{p}" for p in range(self.num_players)]
        # Simplified subkeys to reduce clutter as requested by user
        test_subkeys = ["avg"] + player_subkeys

        for key in stat_keys:
            if key not in self.stats.stats:
                if "test_score" in key or "_score" in key:
                    self.stats._init_key(key, subkeys=test_subkeys)
                else:
                    self.stats._init_key(key)

        # Add common plot types
        self.stats.add_plot_types(
            "score",
            PlotType.ROLLING_AVG,
            PlotType.BEST_FIT_LINE,
            rolling_window=100,
            ema_beta=0.6,
        )
        self.stats.add_plot_types(
            "test_score", PlotType.BEST_FIT_LINE, PlotType.VARIATION_FILL
        )
        self.stats.add_plot_types(
            "episode_length", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "learner_fps", PlotType.ROLLING_AVG, rolling_window=100
        )

        # Test vs agents
        for agent in self.test_agents:
            self.stats.add_plot_types(
                f"vs_{agent.name}_score",
                PlotType.BEST_FIT_LINE,
                PlotType.VARIATION_FILL,
            )
