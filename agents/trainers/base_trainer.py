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

        # Priority: 1. test_agents args, 2. config.game.test_agents, 3. empty list
        self.test_agents = test_agents
        if self.test_agents is None:
            self.test_agents = getattr(config.game, "test_agents", [])

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

        print("Observation dimensions: ", self.obs_dim)
        print("Observation dtype: ", self.obs_dtype)
        print("Number of actions: ", self.num_actions)
        print("Number of players: ", self.num_players)

    def setup(self):
        """Initializes stats and any other setup required before training."""
        self._setup_stats()

    def setup_tester(self):
        """Initializes the EvaluatorActor via an Executor."""
        from agents.workers.actors import EvaluatorActor
        from agents.environments.adapters import GymAdapter
        from agents.action_selectors.selectors import (
            ArgmaxSelector,
            LegalMovesMaskDecorator,
        )

        # 1. Initialize Executor if not already done
        if self.executor is None:
            from agents.factories.executor import create_executor

            self.executor = create_executor(self.config)

        # 2. Prepare worker args
        adapter_cls = self._get_adapter_class()
        env_factory = self.config.game.env_factory
        # Ensure evaluator respects legal moves!
        selector = LegalMovesMaskDecorator(ArgmaxSelector())

        # Priority: Use SearchPolicySource if available (e.g. MuZero), otherwise standard policy_source
        active_policy_source = getattr(self, "search_policy_source", self.policy_source)

        worker_args = (
            adapter_cls,
            (env_factory,),
            self.agent_network,
            active_policy_source,
            None,  # Index 4 (buffer placeholder)
            selector,  # action_selector
            getattr(self.config, "actor_device", "cpu"),
            getattr(self.config.game, "num_actions", None),
            self.num_players,
            self.test_agents,
        )

        # 3. Launch EvaluatorActor
        self.executor.launch_workers(EvaluatorActor, worker_args, num_workers=1)

    def trigger_test(self, state_dict: Dict[str, Any], step: int):
        """
        Triggers evaluation. For LocalExecutor, this runs immediately.
        For TorchMPExecutor, it ensures weights are synced.
        """
        from agents.workers.actors import EvaluatorActor

        if not self._tester_launched:
            self.setup_tester()
            self._tester_launched = True
            if self.executor is None:
                return

        # Update step (weights are shared via shared_memory if multi_process=True)
        self._tester_step = step

        # Signal executor to run test with optional search toggle
        self.executor.request_work(
            EvaluatorActor,
            num_episodes=self.test_trials,
            use_search=getattr(self.config, "eval_use_search", True),
        )

        # If local, run synchronously now
        if not self.config.multi_process:
            results, _ = self.executor.collect_data(
                num_steps=None, worker_type=EvaluatorActor
            )
            for res in results:
                self._process_test_results(res, step)

    def poll_test(self):
        """Polls for background test results from the executor."""
        from agents.workers.actors import EvaluatorActor

        if self.executor is None or not self.config.multi_process:
            return

        # Fetch whatever is available in the result queue for EvaluatorActor
        results, _ = self.executor.collect_data(
            num_steps=None, worker_type=EvaluatorActor
        )
        if results:
            # We only care about the most recent test result for logging
            # The result from EvaluatorActor.evaluate is a dict with 'score', etc.
            # BaseExecutor.collect_data returns (worker_type_name, result_data) tuples for each worker
            for res in results:
                self._process_test_results(res, self._tester_step)

    def stop_test(self):
        """Stops the executor (stops everything)."""
        if self.executor is not None:
            self.executor.stop()
            self.executor = None

    def _process_test_results(self, res: Dict[str, Any], step: int):
        """Logs results from EvaluatorActor."""
        for key, value in res.items():
            if key == "score":
                self.stats.append("test_score", value, subkey="avg")
                print(f"[test] score: {value:.3f} (step {step})")
            elif key == "avg_length":
                self.stats.append("episode_length", value, subkey="test")
                print(f"[test] avg_length: {value:.1f}")
            elif key.startswith("vs_"):
                # Structured results for Matrix Evaluation:
                # f"vs_{opp_name}_score": {"p0": score0, "p1": score1, "avg": avg_score}
                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        self.stats.append(key, subval, subkey=subkey)
                    if "avg" in value:
                        print(f"[test] {key}: {value['avg']:.3f}")
                else:
                    self.stats.append(key, value, subkey="avg")
                    print(f"[test] {key}: {value:.3f}")

    def _get_adapter_class(self):
        """Dynamically selects the correct environment adapter."""
        from agents.environments.adapters import (
            GymAdapter,
            PettingZooAdapter,
            VectorAdapter,
        )

        # If the config specifies vectorized (e.g., PufferLib or Gym.vector)
        if hasattr(self.config, "game") and getattr(
            self.config.game, "vectorized", False
        ):
            return VectorAdapter

        # If there are multiple players, it's PettingZoo (AEC or Parallel)
        if self.num_players > 1:
            return PettingZooAdapter

        # Fallback to standard Gym
        return GymAdapter

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
        torch.save(self.learner.state_dict(), weights_path)

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

    def _record_learner_metrics(self, metric_bundle: Optional[Dict[str, Any]]) -> None:
        if not metric_bundle:
            return

        for key, value in metric_bundle.items():
            if key == "metrics":
                self._record_structured_metrics(value)
            else:
                # Ensure value is detached and converted to float if it's a scalar tensor
                val = value.detach().cpu().item() if torch.is_tensor(value) else value
                self.stats.append(key, val)

    def _record_structured_metrics(self, metrics: Optional[Dict[str, Any]]) -> None:
        if not metrics:
            return

        for key, value in metrics.items():
            if key == "_latent_visualizations":
                for viz_key, viz_payload in value.items():
                    self.stats.add_latent_visualization(
                        viz_key,
                        viz_payload["latents"],
                        labels=viz_payload.get("labels"),
                        method=viz_payload.get("method", "pca"),
                        **viz_payload.get("kwargs", {}),
                    )
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    # Ensure subvalue is detached and converted to float if it's a scalar tensor
                    val = (
                        subvalue.detach().cpu().item()
                        if torch.is_tensor(subvalue)
                        else subvalue
                    )
                    self.stats.set(key, val, subkey=subkey)
            else:
                # Ensure value is detached and converted to float if it's a scalar tensor
                val = value.detach().cpu().item() if torch.is_tensor(value) else value
                self.stats.append(key, val)

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
        learner_state = torch.load(
            weights_path, map_location=device, weights_only=False
        )
        trainer.learner.load_state_dict(learner_state)

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

        if self.num_players > 1:
            for p in range(self.num_players):
                stat_keys.append(f"avg_score_p{p}")

        # Add test_score vs other agents if applicable
        for agent in self.test_agents:
            stat_keys.append(f"vs_{agent.name}_score")

        # Initialize keys
        player_subkeys = [f"p{p}" for p in range(self.num_players)]
        # Simplified subkeys to reduce clutter as requested by user
        test_subkeys = ["avg"] + player_subkeys

        for key in stat_keys:
            if key not in self.stats.stats:
                if (
                    "test_score" in key
                    or "_score" in key
                    or (key == "score" and self.num_players > 1)
                ):
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

    def _record_collection_metrics(self, results: List[Dict[str, Any]]):
        """Standardized recording of collection metrics (score, lengths) from workers."""
        for res in results:
            for score in res.get("batch_scores", []):
                # If plural players, record as subkeys
                if isinstance(score, (list, np.ndarray)) and len(score) > 1:
                    data = {f"p{i}": float(s) for i, s in enumerate(score)}
                    data["avg"] = float(np.mean(score))
                    self.stats.append("score", data)
                else:
                    # Single player or scalar score
                    self.stats.append("score", float(score))
            
            for length in res.get("batch_lengths", []):
                self.stats.append("episode_length", float(length))

            # 2. Worker summaries and rolling stats
            for key, value in res.items():
                if key in ["batch_scores", "batch_lengths"]:
                    continue
                
                # Log avg_score_pX or throughput metrics
                if any(tag in key for tag in ["avg_score", "fps", "steps_per_second"]):
                    if isinstance(value, (int, float, np.number)):
                        self.stats.append(key, float(value))
