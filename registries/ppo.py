import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List, Optional

from modules.agent_nets.modular import ModularAgentNetwork
from modules.backbones.mlp import MLPBackbone
from modules.heads.policy import PolicyHead
from modules.heads.value import ValueHead
from learner.losses.representations import (
    ClassificationRepresentation,
    ScalarRepresentation,
)
from actors.action_selectors.selectors import CategoricalSelector
from actors.action_selectors.decorators import PPODecorator
from actors.action_selectors.policy_sources import NetworkPolicySource

from data.storage.circular import BufferConfig, ModularReplayBuffer
from data.concurrency import LocalBackend
from data.processors import (
    GAEProcessor,
    LegalMovesMaskProcessor,
    AdvantageNormalizer,
    StackedInputProcessor,
)
from data.writers import PPOWriter
from data.samplers.prioritized import WholeBufferSampler

from learner.core import BlackboardEngine
from learner.pipeline.forward_pass import ForwardPassComponent
from learner.losses.optimizer_step import OptimizerStepComponent
from learner.pipeline.components import MetricEarlyStopComponent
from learner.losses.aggregator import LossAggregatorComponent
from learner.losses.policy import ClippedSurrogateLoss
from learner.losses.value import ClippedValueLoss
from learner.pipeline.target_builders import (
    PassThroughTargetComponent,
    TargetFormatterComponent,
    UniversalInfrastructureComponent,
)
from learner.losses.shape_validator import ShapeValidator


def make_ppo_network(
    obs_dim: Tuple[int, ...],
    num_actions: int,
    hidden_widths: List[int] = [64, 64],
    device: torch.device = torch.device("cpu"),
) -> ModularAgentNetwork:
    """
    Creates a standard PPO network with shared or separate MLP backbones.
    """
    return ModularAgentNetwork(
        components={
            "policy_head": PolicyHead(
                input_shape=obs_dim,
                representation=ClassificationRepresentation(num_classes=num_actions),
                neck=MLPBackbone(input_shape=obs_dim, widths=hidden_widths),
            ),
            "value_head": ValueHead(
                input_shape=obs_dim,
                representation=ScalarRepresentation(),
                neck=MLPBackbone(input_shape=obs_dim, widths=hidden_widths),
            ),
        },
    ).to(device)


def make_ppo_replay_buffer(
    obs_dim: Tuple[int, ...],
    num_actions: int,
    steps_per_epoch: int = 512,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> ModularReplayBuffer:
    """
    Creates a standard PPO replay buffer.
    """
    configs = [
        BufferConfig("observations", shape=obs_dim, dtype=torch.float32),
        BufferConfig("actions", shape=(), dtype=torch.int64),
        BufferConfig("rewards", shape=(), dtype=torch.float32),
        BufferConfig("values", shape=(), dtype=torch.float32),
        BufferConfig("old_log_probs", shape=(), dtype=torch.float32),
        BufferConfig("legal_moves_masks", shape=(num_actions,), dtype=torch.bool),
        BufferConfig("advantages", shape=(), dtype=torch.float32),
        BufferConfig("returns", shape=(), dtype=torch.float32),
    ]

    input_stack = StackedInputProcessor(
        [
            GAEProcessor(gamma, gae_lambda),
            LegalMovesMaskProcessor(
                num_actions, input_key="legal_moves", output_key="legal_moves_masks"
            ),
        ]
    )

    return ModularReplayBuffer(
        max_size=steps_per_epoch,
        batch_size=steps_per_epoch,
        buffer_configs=configs,
        input_processor=input_stack,
        output_processor=AdvantageNormalizer(),
        writer=PPOWriter(steps_per_epoch),
        sampler=WholeBufferSampler(),
        backend=LocalBackend(),
    )


def make_ppo_learner(
    agent_network: ModularAgentNetwork,
    optimizer: torch.optim.Optimizer,
    minibatch_size: int,
    num_actions: int,
    device: torch.device,
    clip_param: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: Optional[float] = None,
) -> BlackboardEngine:
    """
    Creates a standard PPO learner (BlackboardEngine).
    """
    pol_rep = agent_network.components["policy_head"].representation
    val_rep = agent_network.components["value_head"].representation

    shape_validator = ShapeValidator(
        minibatch_size=minibatch_size,
        num_actions=num_actions,
        unroll_steps=0,
    )

    policy_loss = ClippedSurrogateLoss(
        clip_param=clip_param,
        entropy_coefficient=entropy_coef,
    )

    value_loss = ClippedValueLoss(
        clip_param=clip_param,
        target_key="returns",
        loss_factor=value_coef,
    )

    components = [
        ForwardPassComponent(agent_network, shape_validator),
        PassThroughTargetComponent(
            ["values", "returns", "actions", "old_log_probs", "advantages"]
        ),
        TargetFormatterComponent({"values": val_rep, "returns": val_rep}),
        UniversalInfrastructureComponent(),
        policy_loss,
        value_loss,
        LossAggregatorComponent(loss_weights={"policy_loss": 1.0, "value_loss": 1.0}),
        OptimizerStepComponent(
            agent_network=agent_network,
            optimizers={"default": optimizer},
            max_grad_norm=max_grad_norm,
        ),
    ]

    if target_kl is not None:
        components.append(MetricEarlyStopComponent(threshold=target_kl))

    return BlackboardEngine(components=components, device=device)
