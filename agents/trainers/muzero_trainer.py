import torch
import time
from typing import Optional, List, Dict, Any
from agents.trainers.base_trainer import BaseTrainer
from agents.factories.learner import build_universal_learner
from agents.learner.batch_iterators import SingleBatchIterator
from agents.factories.replay_buffer import create_muzero_buffer
from agents.action_selectors.selectors import CategoricalSelector
from agents.action_selectors.decorators import TemperatureSelector
from modules.models.agent_network import AgentNetwork
from stats.stats import StatTracker, PlotType


class MuZeroTrainer(BaseTrainer):
    """
    MuZeroTrainer orchestrates the training process for MuZero.
    """

    def __init__(
        self,
        config,
        env,
        device: torch.device,
        name: str = "agent",
        stats: Optional[StatTracker] = None,
        test_agents: List = None,
    ):
        super().__init__(config, env, device, name, stats, test_agents)
        print(f"Trainer Config Unroll Steps: {config.unroll_steps}")
        print(f"Trainer Config Batch Size: {config.minibatch_size}")

        # Create player_id_mapping for multi-player games
        if hasattr(env, "possible_agents"):
            self.player_id_mapping = {
                agent_id: i for i, agent_id in enumerate(env.possible_agents)
            }
        else:
            self.player_id_mapping = {"player_0": 0}

        # 1. Initialize Network
        from agents.factories.builders import make_backbone_fn, make_head_fn
        from functools import partial
        from modules.models.world_model import WorldModel

        # Build functional components
        representation_fn = make_backbone_fn(getattr(config, "representation_backbone", None))
        
        # World Model setup
        wm_cfg = config.world_model
        env_head_fns = {
            name: make_head_fn(h_cfg)
            for name, h_cfg in wm_cfg.env_heads.items()
        }
        
        world_model_fn = partial(
            WorldModel,
            stochastic=getattr(wm_cfg, "stochastic", False),
            num_chance=getattr(wm_cfg, "num_chance", 0),
            observation_shape=getattr(wm_cfg.game, "observation_shape", None),
            use_true_chance_codes=getattr(wm_cfg, "use_true_chance_codes", False),
            env_head_fns=env_head_fns,
            dynamics_fn=make_backbone_fn(wm_cfg.dynamics_backbone),
            afterstate_dynamics_fn=make_backbone_fn(getattr(wm_cfg, "afterstate_dynamics_backbone", None)),
            sigma_head_fn=make_head_fn(getattr(wm_cfg, "chance_probability_head", None)),
            encoder_fn=make_backbone_fn(getattr(wm_cfg, "chance_encoder_backbone", None)),
            action_embedding_dim=getattr(wm_cfg, "action_embedding_dim", 16),
        )

        head_fns = {}
        for name, h_cfg in config.heads.items():
            head_fns[name] = make_head_fn(h_cfg)
        
        # Ensure projector is included if specified separately
        if hasattr(config, "projector") and config.projector:
            head_fns["projector"] = make_head_fn(config.projector)

        self.agent_network = AgentNetwork(
            input_shape=self.obs_dim,
            num_actions=self.num_actions,
            representation_fn=representation_fn,
            world_model_fn=world_model_fn,
            head_fns=head_fns,
            stochastic=config.stochastic,
            num_players=config.game.num_players,
            num_chance_codes=config.num_chance,
        ).to(device)

        if config.kernel_initializer is not None:
            self.agent_network.initialize(config.kernel_initializer)

        if config.multi_process:
            self.agent_network.share_memory()

        # 2. Initialize Action Selector (MCTS)
        from agents.action_selectors.selectors import LegalMovesMaskDecorator
        
        # We wrap with LegalMovesMaskDecorator first, then Temperature
        inner_selector = LegalMovesMaskDecorator(CategoricalSelector())
        self.action_selector = TemperatureSelector(
            inner_selector=inner_selector,
            schedule_config=config.temperature_schedule,
        )

        # 3. The Facts (Replay Buffer)
        self.buffer = create_muzero_buffer(
            observation_dimensions=self.obs_dim,
            max_size=config.replay_buffer_size,
            num_actions=self.num_actions,
            num_players=config.game.num_players,
            player_id_mapping=self.player_id_mapping,
            unroll_steps=config.unroll_steps,
            n_step=config.n_step,
            gamma=config.discount_factor,
            batch_size=config.minibatch_size,
            observation_dtype=self.obs_dtype,
            alpha=config.per_alpha,
            beta=config.per_beta_schedule.initial,
            epsilon=config.per_epsilon,
            use_batch_weights=config.per_use_batch_weights,
            use_initial_max_priority=config.per_use_initial_max_priority,
            lstm_horizon_len=config.lstm_horizon_len,
            value_prefix=config.use_value_prefix,
            tau=config.reanalyze_tau,
            multi_process=config.multi_process,
            observation_quantization=config.observation_quantization,
            observation_compression=config.observation_compression,
        )
        # 5. Initialize Executor
        from agents.factories.executor import create_executor
        self.executor = create_executor(config)

        # 7. The Orchestrator (Universal Learner)
        self.learner = build_universal_learner(
            config=config,
            agent_network=self.agent_network,
            device=device,
            priority_update_fn=self.buffer.update_priorities,
            weight_broadcast_fn=self.executor.update_parameters,
            validator_params={
                "minibatch_size": config.minibatch_size,
                "unroll_steps": config.unroll_steps,
                "num_actions": self.num_actions,
                "num_players": getattr(config.game, "num_players", 1),
                "atom_size": (
                    (config.support_range * 2) + 1
                    if hasattr(config, "support_range") and config.support_range
                    else 1
                ),
            },
        )

        if config.multi_process:
            self.buffer.share_memory()

        # 8. Initialize Workers
        from agents.workers.actors import RolloutActor
        from agents.workers.specialized_actors import ReanalyzeActor
        from agents.environments.adapters import GymAdapter, VectorAdapter
        from agents.action_selectors.policy_sources import NetworkPolicySource, SearchPolicySource

        # Policy sources
        self.policy_source = NetworkPolicySource(self.agent_network)
        # MuZero uses MCTS for rollout
        from agents.factories.search import SearchBackendFactory
        search_engine = SearchBackendFactory.create(config, device=self.device, num_actions=self.num_actions)
        self.search_policy_source = SearchPolicySource(
            search_engine=search_engine,
            agent_network=self.agent_network,
        )

        # Decide between single and vector adapter
        adapter_cls = self._get_adapter_class()
        env_factory = config.game.env_factory
        adapter_args = (env_factory,)

        # Rollout Worker Args
        rollout_args = (
            adapter_cls,
            adapter_args,
            self.agent_network,
            self.search_policy_source,
            self.buffer,
            self.action_selector,
            getattr(config, "actor_device", "cpu"),
            getattr(config.game, "num_actions", None),
            getattr(config.game, "num_players", 1),
            None,   # test_agents
            False,  # flush_incomplete: MuZero stores only complete episodes
        )
        
        num_rollout_workers = config.num_workers if hasattr(config, "num_workers") else 1
        self.executor.launch_workers(RolloutActor, rollout_args, num_workers=num_rollout_workers)

        # Reanalyze Worker Args (if enabled)
        if getattr(config, "use_reanalyze", False):
            reanalyze_args = (
                self.agent_network,
                self.search_policy_source,
                self.buffer,
                config,
            )
            num_reanalyze_workers = getattr(config, "num_reanalyze_workers", 1)
            self.executor.launch_workers(ReanalyzeActor, reanalyze_args, num_workers=num_reanalyze_workers)

        # 6. Compile network for the learner (main process)
        if config.compilation.enabled:
            if device.type == "mps":
                print("Skipping torch.compile on Apple Silicon (MPS).")
            else:
                self.agent_network = torch.compile(
                    self.agent_network,
                    mode=config.compilation.mode,
                    fullgraph=config.compilation.fullgraph,
                )

    def train(self):
        """Main training loop."""
        self._setup_stats()

        print(f"Starting training for {self.config.training_steps} steps...")
        start_time = time.time()

        while self.training_step < self.config.training_steps:
            self.train_step()

            # Poll for background test results
            self.poll_test()

            # Periodic logging
            if self.training_step % 100 == 0 and self.training_step > 0:
                print(f"Step {self.training_step}")

            # Drain stats queue to avoid deadlock
            if hasattr(self.stats, "drain_queue"):
                self.stats.drain_queue()

        self.stop_test()
        self._save_checkpoint()
        print("Training finished.")

    def train_step(self) -> Dict[str, Any]:
        """Perform one training step (batch of gradients)."""
        # 1. Update weights and trigger work
        self.executor.update_parameters(weights=self.agent_network.state_dict())
        
        # 2. Collect data via actor
        from agents.workers.actors import RolloutActor
        from agents.workers.specialized_actors import ReanalyzeActor
        
        # Request re-analysis if enabled (non-blocking if num_steps is None, 
        # but here we might want to trigger it)
        if getattr(self.config, "use_reanalyze", False):
            self.executor.request_work(ReanalyzeActor, batch_size=self.config.minibatch_size)

        # Collect 'replay_interval' steps
        results, collection_stats = self.executor.collect_data(
            num_steps=getattr(self.config, "replay_interval", 1),
            worker_type=RolloutActor
        )

        # 3. Log collection stats
        for res in results:
            for score in res.get("batch_scores", []):
                self.stats.append("score", float(score))
            for length in res.get("batch_lengths", []):
                self.stats.append("episode_length", float(length))

        # 3. Learning step
        if self.buffer.size >= self.config.min_replay_buffer_size:
            for _ in range(self.config.num_minibatches):
                iterator = SingleBatchIterator(self.buffer, self.device)
                for step_metrics in self.learner.step(iterator):
                    self._record_learner_metrics(step_metrics)

            self.training_step += 1

            # 4. Periodic Weight Synchronization handled in train_step or via callback
            # self.executor.update_weights handled at start of train_step now.

            # 5. Periodic checkpointing
            if self.training_step % self.checkpoint_interval == 0:
                self._save_checkpoint()

            # 6. Periodic testing
            if self.training_step % self.test_interval == 0:
                self.trigger_test(self.agent_network.state_dict(), self.training_step)

        return {}

    def _save_checkpoint(self) -> None:
        """Saves MuZero checkpoint."""
        super()._save_checkpoint({})

    def _setup_stats(self):
        """Initializes the stat tracker with all required keys and plot types."""
        from stats.stats import PlotType

        test_score_keys = (
            [f"vs_{agent.name}_score" for agent in self.test_agents]
            if hasattr(self, "test_agents")
            else []
        )

        stat_keys = [
            "score",
            "policy_loss",
            "value_loss",
            "reward_loss",
            "to_play_loss",
            "cons_loss",
            "loss",
            "test_score",
            "episode_length",
            "policy_entropy",
            "value_diff",
            "policy_improvement",
            "learner_fps",
            "actor_fps",
            "mcts_sps",
            "chance_probs",
        ] + test_score_keys

        # Initialize keys
        for key in stat_keys:
            if key not in self.stats.stats:
                if key == "test_score":
                    self.stats._init_key(
                        key
                    )  # Removed min/max subkeys as they are no longer logged
                elif "_score" in key:
                    # Player-specific subkeys for vs_agent tests
                    num_players = self.config.game.num_players
                    subkeys = [f"p{i}" for i in range(num_players)] + ["avg"]
                    self.stats._init_key(key, subkeys=subkeys)
                else:
                    self.stats._init_key(key)

        # Add plot types
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
        if test_score_keys:
            for key in test_score_keys:
                self.stats.add_plot_types(
                    key, PlotType.BEST_FIT_LINE, PlotType.VARIATION_FILL
                )
        self.stats.add_plot_types(
            "policy_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "value_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "reward_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "to_play_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types("cons_loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types(
            "episode_length", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "policy_entropy", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "value_diff", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types("policy_improvement", PlotType.BAR)
        self.stats.add_plot_types(
            "learner_fps", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types("actor_fps", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("mcts_sps", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("chance_probs", PlotType.BAR)
