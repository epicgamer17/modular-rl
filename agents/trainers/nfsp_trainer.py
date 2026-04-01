from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch

from agents.learner.base import UniversalLearner
from agents.learner.batch_iterators import RepeatSampleIterator
from agents.learner.callbacks import (
    PriorityUpdaterCallback,
    ResetNoiseCallback,
    TargetNetworkSyncCallback,
)
from agents.learner.target_builders import (
    TemporalDifferenceBuilder,
    PassThroughTargetBuilder,
)
from agents.learner.losses import (
    ImitationLoss,
    LossPipeline,
    QBootstrappingLoss,
)
from replay_buffers.buffer_factories import (
    create_dqn_buffer,
    create_nfsp_buffer,
)
from agents.action_selectors.policy_sources import NFSPNetworkPolicySource
from agents.action_selectors.selectors import (
    CategoricalSelector,
    EpsilonGreedySelector,
    NFSPSelector,
    ArgmaxSelector,
)
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from agents.executors.local_executor import LocalExecutor
from agents.executors.torch_mp_executor import TorchMPExecutor
from agents.trainers.base_trainer import BaseTrainer
from agents.workers.actors import GymActor, PettingZooActor
from modules.agent_nets.modular import ModularAgentNetwork
from modules.agent_nets.factory import build_modular_agent_network
from modules.utils import get_clean_state_dict
from replay_buffers.sequence import Sequence
from stats.stats import StatTracker, PlotType
from utils.schedule import create_schedule


class _NFSPActorMixin:
    """Mixin to override BaseActor.play_sequence for NFSP.

    Returns a full `Sequence` to the trainer (does not store to any buffer).
    Stores `policy_used` (\"best_response\" / \"average_strategy\") into
    `Sequence.policy_history` so the trainer can route SL storage.
    """

    average_agent_network: ModularAgentNetwork

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        if not params_dict:
            return

        avg_state = params_dict.get("avg_state_dict")
        if avg_state is not None:
            self.average_agent_network.load_state_dict(avg_state)

        if params_dict.get("reset_noise"):
            if hasattr(self.average_agent_network, "reset_noise"):
                self.average_agent_network.reset_noise()

        super().update_parameters(params_dict)

    def _map_all_player_rewards(
        self, rewards: Optional[Dict[Any, float]]
    ) -> Optional[Dict[int, float]]:
        if rewards is None:
            return None

        if not hasattr(self.env, "possible_agents") or not self.env.possible_agents:
            # Single-agent Gym; keep as-is if already numeric.
            if all(isinstance(k, int) for k in rewards.keys()):
                return {int(k): float(v) for k, v in rewards.items()}
            return None

        mapping = {name: idx for idx, name in enumerate(self.env.possible_agents)}
        out: Dict[int, float] = {}
        for k, v in rewards.items():
            if isinstance(k, int):
                out[int(k)] = float(v)
            else:
                if k in mapping:
                    out[mapping[k]] = float(v)
        return out

    @torch.inference_mode()
    def play_sequence(self, stats_tracker: Optional[Any] = None) -> Sequence:
        start_time = time.time()
        sequence = Sequence(self.num_players)

        state, info = self.reset()
        legal_moves = info.get("legal_moves", []) if info else []
        sequence.append(
            state,
            terminated=False,
            truncated=False,
            legal_moves=legal_moves,
            all_player_rewards={},
        )

        while not self._done:
            player_id = self._get_player_id()
            transition = self.step()

            metadata = transition.get("metadata", {})
            policy_used = metadata.get("policy_used", "average_strategy")

            next_info = transition.get("next_info") or {}
            next_legal_moves = next_info.get("legal_moves", [])
            all_player_rewards = self._map_all_player_rewards(
                next_info.get("all_player_rewards")
            )

            sequence.append(
                observation=transition["next_state"],
                terminated=transition["terminated"],
                truncated=transition["truncated"],
                action=transition["action"],
                reward=transition["reward"],
                policy=policy_used,
                value=metadata.get("value"),
                player_id=player_id,
                legal_moves=next_legal_moves,
                all_player_rewards=all_player_rewards,
            )

        sequence.duration_seconds = time.time() - start_time
        self._finalize_episode_info(sequence)
        return sequence


class NFSPGymActor(_NFSPActorMixin, GymActor):
    def __init__(
        self,
        env_factory,
        best_response_agent_network: ModularAgentNetwork,
        average_agent_network: ModularAgentNetwork,
        replay_buffer,
        num_players: int,
        device: torch.device,
        name: str = "agent",
        eta: float = 0.1,
        *,
        worker_id: int = 0,
        # Explicit compilation/video args
        compilation_enabled: bool = False,
        compilation_mode: str = "default",
        compilation_fullgraph: bool = False,
        record_video: bool = False,
        record_video_interval: int = 1000,
    ):
        self.average_agent_network = average_agent_network

        br_selector = EpsilonGreedySelector(epsilon=0.0)
        avg_selector = CategoricalSelector(exploration=True)
        selector = NFSPSelector(
            br_selector=br_selector, avg_selector=avg_selector, eta=eta
        )
        policy_source = NFSPNetworkPolicySource(
            best_response_network=best_response_agent_network,
            average_network=average_agent_network,
        )
        super().__init__(
            env_factory=env_factory,
            agent_network=best_response_agent_network,
            action_selector=selector,
            replay_buffer=replay_buffer,
            num_players=num_players,
            device=device,
            name=name,
            worker_id=worker_id,
            policy_source=policy_source,
            compilation_enabled=compilation_enabled,
            compilation_mode=compilation_mode,
            compilation_fullgraph=compilation_fullgraph,
            record_video=record_video,
            record_video_interval=record_video_interval,
        )


class NFSPPettingZooActor(_NFSPActorMixin, PettingZooActor):
    def __init__(
        self,
        env_factory,
        best_response_agent_network: ModularAgentNetwork,
        average_agent_network: ModularAgentNetwork,
        replay_buffer,
        num_players: int,
        device: torch.device,
        name: str = "agent",
        eta: float = 0.1,
        *,
        worker_id: int = 0,
        # Explicit compilation/video args
        compilation_enabled: bool = False,
        compilation_mode: str = "default",
        compilation_fullgraph: bool = False,
        record_video: bool = False,
        record_video_interval: int = 1000,
    ):
        self.average_agent_network = average_agent_network

        br_selector = EpsilonGreedySelector(epsilon=0.0)
        avg_selector = CategoricalSelector(exploration=True)
        selector = NFSPSelector(
            br_selector=br_selector, avg_selector=avg_selector, eta=eta
        )
        policy_source = NFSPNetworkPolicySource(
            best_response_network=best_response_agent_network,
            average_network=average_agent_network,
        )
        super().__init__(
            env_factory=env_factory,
            agent_network=best_response_agent_network,
            action_selector=selector,
            replay_buffer=replay_buffer,
            num_players=num_players,
            device=device,
            name=name,
            worker_id=worker_id,
            policy_source=policy_source,
            compilation_enabled=compilation_enabled,
            compilation_mode=compilation_mode,
            compilation_fullgraph=compilation_fullgraph,
            record_video=record_video,
            record_video_interval=record_video_interval,
        )


def _pick_nfsp_actor(env: Any) -> type[GymActor] | type[PettingZooActor]:
    is_pz = hasattr(env, "possible_agents")
    if not is_pz and hasattr(env, "unwrapped"):
        is_pz = hasattr(env.unwrapped, "possible_agents")
    return NFSPPettingZooActor if is_pz else NFSPGymActor


class NFSPTrainer(BaseTrainer):
    """NFSP trainer using UniversalLearner for RL/SL updates."""

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

        if not getattr(config, "shared_networks_and_buffers", True):
            raise NotImplementedError(
                "NFSPTrainer currently supports shared_networks_and_buffers=True only."
            )

        self.player_ids = list(range(self.num_players))

        self.rl_config = config.rl_configs[0]
        self.sl_config = config.sl_configs[0]
        rl_config = self.rl_config
        sl_config = self.sl_config

        # Networks (Best Response + Target, and Average Strategy)
        self.br_agent_network = build_modular_agent_network(
            config=rl_config,
            input_shape=self.obs_dim,
            num_actions=self.num_actions,
        ).to(device)
        self.br_target_agent_network = build_modular_agent_network(
            config=rl_config,
            input_shape=self.obs_dim,
            num_actions=self.num_actions,
        ).to(device)
        self.br_target_agent_network.load_state_dict(
            get_clean_state_dict(self.br_agent_network), strict=False
        )

        self.avg_agent_network = build_modular_agent_network(
            config=sl_config,
            input_shape=self.obs_dim,
            num_actions=self.num_actions,
        ).to(device)

        if self.config.multi_process:
            self.br_agent_network.share_memory()
            self.br_target_agent_network.share_memory()
            self.avg_agent_network.share_memory()

        # 1) RL buffer + learner (Best Response)
        self.rl_buffer = create_dqn_buffer(
            observation_dimensions=self.obs_dim,
            max_size=rl_config.replay_buffer_size,
            num_actions=self.num_actions,
            batch_size=rl_config.minibatch_size,
            observation_dtype=self.obs_dtype,
            config=rl_config,
        )
        self.rl_per_beta_schedule = create_schedule(
            getattr(rl_config, "per_beta_schedule", None)
        )

        def create_opt(params, sub_config):
            if sub_config.optimizer == Adam:
                return Adam(
                    params=params,
                    lr=sub_config.learning_rate,
                    eps=sub_config.adam_epsilon,
                    weight_decay=sub_config.weight_decay,
                )
            elif sub_config.optimizer == SGD:
                return SGD(
                    params=params,
                    lr=sub_config.learning_rate,
                    momentum=sub_config.momentum,
                    weight_decay=sub_config.weight_decay,
                )
            else:
                raise ValueError(f"Unsupported optimizer: {sub_config.optimizer}")

        rl_optimizer = create_opt(self.br_agent_network.parameters(), rl_config)
        rl_scheduler = get_lr_scheduler(rl_optimizer, rl_config)
        from agents.learner.target_builders import (
            DistributionalTargetBuilder,
        )

        is_distributional = getattr(rl_config, "atom_size", 1) > 1
        builder_cls = (
            DistributionalTargetBuilder
            if is_distributional
            else TemporalDifferenceBuilder
        )

        rl_target_builder = builder_cls(
            target_network=self.br_target_agent_network,
            gamma=rl_config.discount_factor,
            n_step=rl_config.n_step,
            bootstrap_on_truncated=getattr(rl_config, "bootstrap_on_truncated", False),
        )

        rl_rep = self.br_agent_network.components["q_head"].representation
        td_loss_module = QBootstrappingLoss(
            device=device,
            representation=rl_rep,
            is_categorical=is_distributional,
            loss_fn=getattr(rl_config, "loss_function", None),
        )
        rl_loss_pipeline = LossPipeline(
            modules=[td_loss_module],
            minibatch_size=rl_config.minibatch_size,
            num_actions=self.num_actions,
            atom_size=getattr(rl_config, "atom_size", 1),
            unroll_steps=0,  # NFSP RL is DQ(N)
        )

        from agents.learner.losses.shape_validator import ShapeValidator

        rl_shape_validator = ShapeValidator(
            minibatch_size=rl_config.minibatch_size,
            num_actions=self.num_actions,
            unroll_steps=0,
            atom_size=getattr(rl_config, "atom_size", 1),
        )

        self.rl_learner = UniversalLearner(
            agent_network=self.br_agent_network,
            device=device,
            num_actions=self.num_actions,
            observation_dimensions=self.obs_dim,
            observation_dtype=self.obs_dtype,
            target_builder=rl_target_builder,
            loss_pipeline=rl_loss_pipeline,
            optimizer=rl_optimizer,
            lr_scheduler=rl_scheduler,
            shape_validator=rl_shape_validator,
            clipnorm=getattr(rl_config, "clipnorm", None),
            gradient_accumulation_steps=getattr(
                rl_config, "gradient_accumulation_steps", 1
            ),
            max_grad_norm=getattr(rl_config, "max_grad_norm", None),
            callbacks=[
                TargetNetworkSyncCallback(
                    target_network=self.br_target_agent_network,
                    sync_interval=rl_config.transfer_interval,
                    soft_update=getattr(rl_config, "soft_update", False),
                    ema_beta=getattr(rl_config, "ema_beta", 0.99),
                ),
                ResetNoiseCallback(),
                PriorityUpdaterCallback(
                    priority_update_fn=self.rl_buffer.update_priorities,
                    set_beta_fn=self.rl_buffer.set_beta,
                    per_beta_schedule=self.rl_per_beta_schedule,
                ),
            ],
        )

        # 2) SL buffer + learner (Average Strategy)
        self.sl_buffer = create_nfsp_buffer(
            observation_dimensions=self.obs_dim,
            max_size=sl_config.replay_buffer_size,
            num_actions=self.num_actions,
            batch_size=sl_config.minibatch_size,
            observation_dtype=self.obs_dtype,
        )

        sl_optimizer = create_opt(self.avg_agent_network.parameters(), sl_config)
        sl_scheduler = get_lr_scheduler(sl_optimizer, sl_config)
        sl_rep = self.avg_agent_network.components["policy_head"].representation
        sl_loss_pipeline = LossPipeline(
            modules=[
                ImitationLoss(
                    device=device,
                    representation=sl_rep,
                    loss_fn=sl_config.policy_loss_function,
                    loss_factor=getattr(sl_config, "policy_loss_factor", 1.0),
                )
            ],
            minibatch_size=sl_config.minibatch_size,
            num_actions=self.num_actions,
            unroll_steps=0,
        )

        sl_shape_validator = ShapeValidator(
            minibatch_size=sl_config.minibatch_size,
            num_actions=self.num_actions,
            unroll_steps=0,
        )

        # SL uses PassThroughTargetBuilder for target_policies -> policies mapping if needed,
        # or just keeping target_policies
        sl_target_builder = PassThroughTargetBuilder(["target_policies"])

        self.sl_learner = UniversalLearner(
            agent_network=self.avg_agent_network,
            device=device,
            num_actions=self.num_actions,
            observation_dimensions=self.obs_dim,
            observation_dtype=self.obs_dtype,
            target_builder=sl_target_builder,
            loss_pipeline=sl_loss_pipeline,
            optimizer=sl_optimizer,
            lr_scheduler=sl_scheduler,
            shape_validator=sl_shape_validator,
            clipnorm=getattr(sl_config, "clipnorm", None),
            gradient_accumulation_steps=getattr(
                sl_config, "gradient_accumulation_steps", 1
            ),
            max_grad_norm=getattr(sl_config, "max_grad_norm", None),
            callbacks=[ResetNoiseCallback()],
        )

        # Schedules/params for actors (best response exploration)
        self.epsilon_schedule = (
            create_schedule(rl_config.epsilon_schedule)
            if getattr(rl_config, "epsilon_schedule", None) is not None
            else None
        )

        # Executor
        self.executor = (
            TorchMPExecutor() if self.config.multi_process else LocalExecutor()
        )
        self.actor_cls = _pick_nfsp_actor(env)
        worker_args = (
            self.config.game.env_factory,
            self.br_agent_network,
            self.avg_agent_network,
            None,  # replay buffer (trainer stores transitions)
            self.num_players,
            device,
            self.obs_dim,
            self.num_actions,
            self.name,
            getattr(self.config, "anticipatory_param", 0.1),
        )
        worker_kwargs = {
            "compilation_enabled": rl_config.compilation.enabled,
            "compilation_mode": rl_config.compilation.mode,
            "compilation_fullgraph": rl_config.compilation.fullgraph,
            "record_video": getattr(self.config, "record_video", False),
            "record_video_interval": getattr(self.config, "record_video_interval", 1000),
        }
        self.executor.launch(
            self.actor_cls, worker_args, self.config.num_workers, **worker_kwargs
        )

        # BaseTrainer expects a single agent_network/action_selector for tester wiring.
        self.agent_network = self.br_agent_network
        self.action_selector = EpsilonGreedySelector(epsilon=0.0)

    @property
    def current_epsilon(self) -> float:
        if self.epsilon_schedule is None:
            return float(getattr(self.action_selector, "epsilon", 0.0))
        return float(self.epsilon_schedule.get_value(step=self.training_step))

    def train_step(self):
        self._update_worker_weights()

        sequences, collection_stats = self.executor.collect_data(
            min_samples=None,
            worker_type=self.actor_cls,
        )

        for key, val in collection_stats.items():
            self.stats.append(key, val)

        for sequence in sequences:
            self._store_sequence_transitions(sequence)

        for _ in range(self.config.num_minibatches):
            # 1. RL Step
            if self.rl_buffer.size >= self.rl_config.min_replay_buffer_size:
                if self.rl_per_beta_schedule is not None:
                    self.rl_per_beta_schedule.step()

                rl_iter = RepeatSampleIterator(
                    self.rl_buffer,
                    self.rl_config.training_iterations,
                    self.device,
                )
                for rl_metrics in self.rl_learner.step(batch_iterator=rl_iter):
                    self._record_learner_metrics(
                        self._prefix_metric_bundle("rl", rl_metrics)
                    )

            # 2. SL Step (supervised/imitation)
            if self.sl_buffer.size >= self.sl_config.min_replay_buffer_size:
                sl_iter = RepeatSampleIterator(
                    self.sl_buffer,
                    self.sl_config.training_iterations,
                    self.device,
                )
                for sl_metrics in self.sl_learner.step(batch_iterator=sl_iter):
                    self._record_learner_metrics(
                        self._prefix_metric_bundle("sl", sl_metrics)
                    )

        self.training_step += 1

    def _update_worker_weights(self):
        params = {
            "eta": float(getattr(self.config, "anticipatory_param", 0.1)),
            "epsilon": self.current_epsilon,
        }

        if not self.config.multi_process:
            params["avg_state_dict"] = self.avg_agent_network.state_dict()

        self.executor.update_weights(self.br_agent_network.state_dict(), params=params)

    def _store_sequence_transitions(self, sequence: Sequence):
        if not sequence.action_history:
            return

        player_rewards = (
            self._compute_player_rewards(sequence) if self.num_players > 1 else None
        )

        for i in range(len(sequence.action_history)):
            obs = sequence.observation_history[i]
            legal_moves = (
                sequence.legal_moves_history[i]
                if i < len(sequence.legal_moves_history)
                else []
            )
            action = sequence.action_history[i]

            player_id = (
                sequence.player_id_history[i]
                if sequence.player_id_history and i < len(sequence.player_id_history)
                else 0
            )

            # Find next turn index for this player to get their next observation.
            next_turn_idx = -1
            if self.num_players > 1 and sequence.player_id_history:
                for j in range(i + 1, len(sequence.action_history)):
                    if sequence.player_id_history[j] == player_id:
                        next_turn_idx = j
                        break

            if next_turn_idx != -1:
                next_obs = sequence.observation_history[next_turn_idx]
                next_legal_moves = (
                    sequence.legal_moves_history[next_turn_idx]
                    if next_turn_idx < len(sequence.legal_moves_history)
                    else []
                )
                done = False
            else:
                next_obs = sequence.observation_history[-1]
                next_legal_moves = (
                    sequence.legal_moves_history[-1]
                    if sequence.legal_moves_history
                    else []
                )
                done = True

            if player_rewards is not None and player_id in player_rewards:
                reward = player_rewards[player_id].get(i, 0.0)
            else:
                reward = sequence.rewards[i] if i < len(sequence.rewards) else 0.0

            policy_used = (
                sequence.policy_history[i]
                if i < len(sequence.policy_history)
                else "average_strategy"
            )

            # Always store in RL replay buffer
            self.rl_buffer.store(
                observations=obs,
                legal_moves=legal_moves,
                actions=action,
                rewards=reward,
                next_observations=next_obs,
                next_legal_moves=next_legal_moves,
                terminated=done,
                truncated=False,
                dones=done,
            )

            # If best_response was used, store in SL reservoir buffer
            if policy_used == "best_response":
                target_policy = torch.zeros(self.num_actions, dtype=torch.float32)
                target_policy[action] = 1.0
                self.sl_buffer.store(
                    observations=obs,
                    legal_moves=legal_moves,
                    target_policies=target_policy,
                )

    def _compute_player_rewards(
        self, sequence: Sequence
    ) -> Dict[int, Dict[int, float]]:
        player_rewards: Dict[int, Dict[int, float]] = {
            pid: {} for pid in self.player_ids
        }
        last_action_idx: Dict[int, int] = {}

        for i in range(len(sequence.action_history)):
            acting_player = (
                sequence.player_id_history[i]
                if sequence.player_id_history and i < len(sequence.player_id_history)
                else 0
            )

            all_rewards = (
                sequence.all_player_rewards_history[i + 1]
                if i + 1 < len(sequence.all_player_rewards_history)
                else {}
            )

            for pid, r in all_rewards.items():
                if pid in last_action_idx:
                    idx = last_action_idx[pid]
                    player_rewards[pid][idx] = player_rewards[pid].get(
                        idx, 0.0
                    ) + float(r)

            last_action_idx[acting_player] = i

        return player_rewards

    def _prefix_metric_bundle(
        self, prefix: str, metric_bundle: Dict[str, Any]
    ) -> Dict[str, Any]:
        prefixed = {
            f"{prefix}_{key}": value
            for key, value in metric_bundle.items()
            if key != "metrics"
        }

        nested_metrics = metric_bundle.get("metrics")
        if nested_metrics:
            prefixed["metrics"] = self._prefix_structured_metrics(
                prefix, nested_metrics
            )

        return prefixed

    def _prefix_structured_metrics(
        self, prefix: str, metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        from copy import deepcopy

        prefixed_metrics: Dict[str, Any] = {}
        for key, value in metrics.items():
            if key == "_latent_visualizations":
                prefixed_metrics[key] = {
                    f"{prefix}_{viz_key}": deepcopy(viz_payload)
                    for viz_key, viz_payload in value.items()
                }
            else:
                prefixed_metrics[f"{prefix}_{key}"] = value
        return prefixed_metrics

    def _save_checkpoint(self) -> None:
        super()._save_checkpoint({})

    def _setup_stats(self) -> None:
        super()._setup_stats()
        for key in ["rl_loss", "sl_loss"]:
            if key not in self.stats.stats:
                self.stats._init_key(key)
        self.stats.add_plot_types("score", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("test_score", PlotType.BEST_FIT_LINE)
