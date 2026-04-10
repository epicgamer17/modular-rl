import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List, Optional

from modules.agent_nets.modular import ModularAgentNetwork
from modules.backbones.mlp import MLPBackbone
from modules.heads.policy import PolicyHead
from modules.heads.value import ValueHead
from modules.representations import (
    ClassificationRepresentation,
    ScalarRepresentation,
)
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

from core import BlackboardEngine
from components.telemetry import TelemetryComponent
from components.neural import ForwardPassComponent
from components.losses import (
    OptimizerStepComponent,
    MetricEarlyStopComponent,
    LossAggregatorComponent,
    ClippedSurrogateLoss,
    ClippedValueLoss,
    ShapeValidator,
)
from components.targets import (
    ScalarFormatterComponent,
)
from components.environments import GymObservationComponent, GymStepComponent
from components.selectors import (
    NetworkInferenceComponent,
    ActionSelectorComponent,
    PPODecoratorComponent,
)
from components.memory import BufferStoreComponent


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
        BufferConfig("dones", shape=(), dtype=torch.bool),
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
        actions_key="data.actions",
        old_log_probs_key="data.old_log_probs",
        advantages_key="data.advantages",
    )

    value_loss = ClippedValueLoss(
        clip_param=clip_param,
        target_key="returns",
        loss_factor=value_coef,
    )

    components = [
        ForwardPassComponent(agent_network, shape_validator),
        ScalarFormatterComponent(source_key="data.values", dest_key="values", representation=val_rep),
        ScalarFormatterComponent(source_key="data.returns", dest_key="returns", representation=val_rep),
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


def make_ppo_actor_engine(
    env: Any,
    agent_network: ModularAgentNetwork,
    replay_buffer: Optional[ModularReplayBuffer],
    obs_dim: Tuple[int, ...],
    num_actions: int,
    exploration: bool = True,
    device: torch.device = torch.device("cpu"),
) -> BlackboardEngine:
    """
    Creates a standard PPO actor engine.
    """
    obs_component = GymObservationComponent(env)
    step_component = GymStepComponent(env, obs_component)

    components = [
        obs_component,
        NetworkInferenceComponent(agent_network, obs_dim),
        ActionSelectorComponent(
            input_key="logits", temperature=1.0 if exploration else 0.0
        ),
        PPODecoratorComponent(),
        step_component,
        TelemetryComponent(name="ppo_actor"),
    ]

    if replay_buffer is not None:
        # PPO usually stores specific fields
        ppo_field_map = {
            "observations": "data.obs",
            "actions": "meta.action",
            "rewards": "data.reward",
            "dones": "data.done",
            "values": "meta.action_metadata.value",
            "old_log_probs": "meta.action_metadata.log_prob",
            "legal_moves": "data.info.legal_moves",
        }
        components.append(BufferStoreComponent(replay_buffer, field_map=ppo_field_map))

    return BlackboardEngine(components=components, device=device)
