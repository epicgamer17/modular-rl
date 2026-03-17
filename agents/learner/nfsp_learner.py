"""
NFSPLearner handles the training logic for NFSP, coordinating updates for both
the Best Response (RL) network and the Average Strategy (SL) network.
"""

from copy import deepcopy
from typing import Any, Dict, Iterator, Tuple

import torch
import torch.nn as nn

from agents.learners.base import UniversalLearner
from agents.learners.batch_iterators import RepeatSampleIterator
from agents.learners.callbacks import (
    PriorityUpdaterCallback,
    ResetNoiseCallback,
    TargetNetworkSyncCallback,
)
from agents.learners.target_builders import (
    TemporalDifferenceBuilder,
    TDCategoricalProjectionBuilder,
)
from losses.losses import C51Loss, ImitationLoss, LossPipeline, StandardDQNLoss
from modules.utils import get_lr_scheduler
from replay_buffers.buffer_factories import create_dqn_buffer, create_nfsp_buffer
from utils.schedule import create_schedule


class NFSPLearner:
    """
    NFSPLearner manages the dual-learning process of NFSP.
    It composes two UniversalLearners:
    - Best Response (RL): DQN-style targets + TD loss + PER
    - Average Strategy (SL): supervised imitation loss on a reservoir buffer
    """

    def __init__(
        self,
        config,
        best_response_agent_network: nn.Module,
        best_response_target_agent_network: nn.Module,
        average_agent_network: nn.Module,
        device: torch.device,
        num_actions: int,
        observation_dimensions: Tuple[int, ...],
        observation_dtype: torch.dtype,
    ):
        """
        Initializes the NFSPLearner.

        Args:
            config: NFSPConfig with hyperparameters.
            best_response_agent_network: Network for Best Response.
            best_response_target_agent_network: Target network for Best Response.
            average_agent_network: Network for Average Strategy.
            device: Torch device for tensors.
            num_actions: Number of discrete actions.
            observation_dimensions: Shape of observations.
            observation_dtype: Dtype for observations.
        """
        self.config = config
        self.device = device
        self.num_actions = num_actions
        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype
        self.training_step = 0

        rl_config = config.rl_configs[0]
        sl_config = config.sl_configs[0]

        # 1) RL buffer + learner (Best Response)
        self.rl_buffer = create_dqn_buffer(
            observation_dimensions=observation_dimensions,
            max_size=rl_config.replay_buffer_size,
            num_actions=num_actions,
            batch_size=rl_config.minibatch_size,
            observation_dtype=observation_dtype,
            config=rl_config,
        )
        self._rl_per_beta_schedule = create_schedule(
            getattr(rl_config, "per_beta_schedule", None)
        )

        from torch.optim.adam import Adam
        from torch.optim.sgd import SGD

        if rl_config.optimizer == Adam:
            rl_optimizer = rl_config.optimizer(
                params=best_response_agent_network.parameters(),
                lr=rl_config.learning_rate,
                eps=rl_config.adam_epsilon,
                weight_decay=rl_config.weight_decay,
            )
        elif rl_config.optimizer == SGD:
            rl_optimizer = rl_config.optimizer(
                params=best_response_agent_network.parameters(),
                lr=rl_config.learning_rate,
                momentum=rl_config.momentum,
                weight_decay=rl_config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {rl_config.optimizer}")

        rl_scheduler = get_lr_scheduler(rl_optimizer, rl_config)
        if rl_config.atom_size > 1:
            rl_target_builder = TDCategoricalProjectionBuilder(
                target_network=best_response_target_agent_network,
                v_min=rl_config.v_min,
                v_max=rl_config.v_max,
                atom_size=rl_config.atom_size,
                gamma=rl_config.discount_factor,
                n_step=rl_config.n_step,
                bootstrap_on_truncated=getattr(
                    rl_config, "bootstrap_on_truncated", False
                ),
                device=device,
            )
        else:
            rl_target_builder = TemporalDifferenceBuilder(
                target_network=best_response_target_agent_network,
                gamma=rl_config.discount_factor,
                n_step=rl_config.n_step,
                bootstrap_on_truncated=getattr(
                    rl_config, "bootstrap_on_truncated", False
                ),
            )

        from agents.action_selectors.selectors import ArgmaxSelector

        training_selector = ArgmaxSelector()
        td_loss_module = (
            C51Loss(config=rl_config, device=device, action_selector=training_selector)
            if rl_config.atom_size > 1
            else StandardDQNLoss(
                config=rl_config, device=device, action_selector=training_selector
            )
        )
        rl_loss_pipeline = LossPipeline([td_loss_module])

        if rl_config.atom_size > 1:
            rl_loss_pipeline.validate_dependencies(
                network_output_keys={"q_logits"},
                target_keys={"q_logits", "actions"},
            )
        else:
            rl_loss_pipeline.validate_dependencies(
                network_output_keys={"q_values"},
                target_keys={"q_values", "actions"},
            )

        self.rl_learner = UniversalLearner(
            config=rl_config,
            agent_network=best_response_agent_network,
            device=device,
            num_actions=num_actions,
            observation_dimensions=observation_dimensions,
            observation_dtype=observation_dtype,
            target_builder=rl_target_builder,
            loss_pipeline=rl_loss_pipeline,
            optimizer=rl_optimizer,
            lr_scheduler=rl_scheduler,
            clipnorm=rl_config.clipnorm,
            callbacks=[
                TargetNetworkSyncCallback(
                    target_network=best_response_target_agent_network,
                    sync_interval=rl_config.transfer_interval,
                    soft_update=getattr(rl_config, "soft_update", False),
                    ema_beta=getattr(rl_config, "ema_beta", 0.99),
                ),
                ResetNoiseCallback(),
                PriorityUpdaterCallback(
                    priority_update_fn=self.rl_buffer.update_priorities,
                    set_beta_fn=self.rl_buffer.set_beta,
                    per_beta_schedule=self._rl_per_beta_schedule,
                ),
            ],
        )
        self.rl_learner.target_agent_network = best_response_target_agent_network
        self.rl_learner.replay_buffer = self.rl_buffer

        # 2) SL buffer + learner (Average Strategy)
        self.sl_buffer = create_nfsp_buffer(
            observation_dimensions=observation_dimensions,
            max_size=sl_config.replay_buffer_size,
            num_actions=num_actions,
            batch_size=sl_config.minibatch_size,
            observation_dtype=observation_dtype,
        )

        if sl_config.optimizer == Adam:
            sl_optimizer = sl_config.optimizer(
                params=average_agent_network.parameters(),
                lr=sl_config.learning_rate,
                eps=sl_config.adam_epsilon,
                weight_decay=sl_config.weight_decay,
            )
        elif sl_config.optimizer == SGD:
            sl_optimizer = sl_config.optimizer(
                params=average_agent_network.parameters(),
                lr=sl_config.learning_rate,
                momentum=sl_config.momentum,
                weight_decay=sl_config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {sl_config.optimizer}")

        sl_scheduler = get_lr_scheduler(sl_optimizer, sl_config)
        sl_loss_pipeline = LossPipeline([ImitationLoss(sl_config, device, num_actions)])
        sl_loss_pipeline.validate_dependencies(
            network_output_keys={"policies"},
            target_keys={"target_policies"},
        )

        self.sl_learner = UniversalLearner(
            config=sl_config,
            agent_network=average_agent_network,
            device=device,
            num_actions=num_actions,
            observation_dimensions=observation_dimensions,
            observation_dtype=observation_dtype,
            target_builder=None,
            loss_pipeline=sl_loss_pipeline,
            optimizer=sl_optimizer,
            lr_scheduler=sl_scheduler,
            clipnorm=sl_config.clipnorm,
            callbacks=[ResetNoiseCallback()],
        )
        self.sl_learner.replay_buffer = self.sl_buffer

    @property
    def sl_optimizer(self) -> torch.optim.Optimizer:
        """Backwards compatibility: expose SL optimizer from composed learner."""
        return self.sl_learner.optimizers["default"]

    @property
    def sl_replay_buffer(self):
        """Backwards compatibility: expose SL buffer from composed learner."""
        return self.sl_buffer

    def store(
        self,
        observation: Any,
        legal_moves: list,
        action: int,
        reward: float,
        next_observation: Any,
        next_legal_moves: list,
        done: bool,
        policy_used: str,
    ) -> None:
        """
        Stores a transition in the appropriate replay buffers.

        Args:
            observation: Current observation.
            legal_moves: List of legal action indices.
            action: Action taken.
            reward: Reward received.
            next_observation: Next observation.
            next_legal_moves: List of legal action indices for next state.
            done: Whether the episode finished.
            policy_used: Either "best_response" or "average_strategy".
        """
        # Always store in RL replay buffer
        self.rl_buffer.store(
            observations=observation,
            legal_moves=legal_moves,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
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
                observations=observation,
                legal_moves=legal_moves,
                target_policies=target_policy,
            )

    def step(self) -> Iterator[Dict[str, Any]]:
        """
        Performs training steps for both RL and SL components.
        """
        # 1. RL Step
        if self.rl_buffer.size >= self.rl_learner.config.min_replay_buffer_size:
            if self._rl_per_beta_schedule is not None:
                self._rl_per_beta_schedule.step()

            rl_iter = RepeatSampleIterator(
                self.rl_buffer, self.rl_learner.config.training_iterations, self.device
            )
            for rl_metrics in self.rl_learner.step(batch_iterator=rl_iter):
                yield self._prefix_metric_bundle("rl", rl_metrics)

        # 2. SL Step (supervised/imitation)
        if self.sl_replay_buffer.size >= self.sl_learner.config.min_replay_buffer_size:
            sl_iter = RepeatSampleIterator(
                self.sl_buffer, self.sl_learner.config.training_iterations, self.device
            )
            for sl_metrics in self.sl_learner.step(batch_iterator=sl_iter):
                yield self._prefix_metric_bundle("sl", sl_metrics)

        self.training_step += 1

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
