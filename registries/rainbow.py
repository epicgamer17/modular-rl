import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List, Optional

from modules.agent_nets.modular import ModularAgentNetwork
from modules.backbones.mlp import MLPBackbone
from modules.heads.q import QHead, DuelingQHead
from modules.representations import C51Representation
from actors.action_selectors.selectors import ActionSelector

from data.storage.circular import BufferConfig, ModularReplayBuffer, CircularWriter
from data.samplers.prioritized import PrioritizedSampler
from data.concurrency import LocalBackend
from data.processors import (
    StackedInputProcessor,
    NStepInputProcessor,
    TerminationFlagsInputProcessor,
    LegalMovesMaskProcessor,
    FilterKeysInputProcessor,
    StandardOutputProcessor,
)
from data.writers import SharedCircularWriter
from components.environments import GymObservationComponent, GymStepComponent
from components.telemetry import TelemetryComponent
from components.memory import BufferStoreComponent
from components.selectors import NetworkInferenceComponent, ActionSelectorComponent

from core import BlackboardEngine
from components.neural import ForwardPassComponent
from components.losses import OptimizerStepComponent
from components.memory import BetaScheduleComponent
from components.routing import ResetNoiseComponent

from components.losses import LossAggregatorComponent
from components.memory import PriorityUpdateComponent
from components.losses import LossPriorityComponent
from components.losses import QBootstrappingLoss
from components.targets import (
    DistributionalTargetComponent,
)
from utils.schedule import Schedule, ConstantSchedule


def make_rainbow_network(
    obs_dim: Tuple[int, ...],
    num_actions: int,
    hidden_widths: List[int] = [128],
    atom_size: int = 51,
    v_min: float = 0.0,
    v_max: float = 200.0,
    noisy_sigma: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> ModularAgentNetwork:
    """
    Creates a standard Rainbow DQN network (Distributional + Noisy).
    """
    representation = C51Representation(vmin=v_min, vmax=v_max, bins=atom_size)

    backbone = MLPBackbone(input_shape=obs_dim, widths=hidden_widths)
    # Head uses Dueling architecture: [Backbone] -> [Value Stream / Advantage Stream]
    head = DuelingQHead(
        input_shape=backbone.output_shape,
        num_actions=num_actions,
        representation=representation,
        value_hidden_widths=[128],
        advantage_hidden_widths=[128],
        noisy_sigma=noisy_sigma,
    )
    return ModularAgentNetwork(
        components={
            "feature_block": backbone,
            "q_head": head,
        },
        atom_size=atom_size,
    ).to(device)


def make_rainbow_replay_buffer(
    obs_dim: Tuple[int, ...],
    num_actions: int,
    max_size: int = 100000,
    batch_size: int = 128,
    n_step: int = 3,
    gamma: float = 0.99,
    per_alpha: float = 0.2,
    per_beta: float = 0.6,
    per_epsilon: float = 1e-6,
    shared: bool = False,
) -> ModularReplayBuffer:
    """
    Creates a standard Rainbow replay buffer (Prioritized + N-Step).
    """
    buffer_configs = [
        BufferConfig("observations", shape=obs_dim, dtype=torch.float32),
        BufferConfig("actions", shape=(), dtype=torch.int64),
        BufferConfig("rewards", shape=(), dtype=torch.float32),
        BufferConfig("next_observations", shape=obs_dim, dtype=torch.float32),
        BufferConfig("terminated", shape=(), dtype=torch.bool),
        BufferConfig("truncated", shape=(), dtype=torch.bool),
        BufferConfig("dones", shape=(), dtype=torch.bool),
        BufferConfig("next_legal_moves_masks", shape=(num_actions,), dtype=torch.bool),
    ]

    input_processor = StackedInputProcessor(
        [
            TerminationFlagsInputProcessor(),
            NStepInputProcessor(n_step=n_step, gamma=gamma),
            LegalMovesMaskProcessor(
                num_actions,
                input_key="next_legal_moves",
                output_key="next_legal_moves_masks",
            ),
            FilterKeysInputProcessor(
                [
                    "observations",
                    "actions",
                    "rewards",
                    "next_observations",
                    "terminated",
                    "truncated",
                    "dones",
                    "next_legal_moves_masks",
                ]
            ),
        ]
    )

    sampler = PrioritizedSampler(
        max_size=max_size,
        alpha=per_alpha,
        beta=per_beta,
        epsilon=per_epsilon,
    )

    writer_cls = SharedCircularWriter if shared else CircularWriter

    return ModularReplayBuffer(
        max_size=max_size,
        batch_size=batch_size,
        buffer_configs=buffer_configs,
        input_processor=input_processor,
        output_processor=StandardOutputProcessor(),
        writer=writer_cls(max_size=max_size),
        sampler=sampler,
        backend=LocalBackend(),
    )


def make_rainbow_actor_engine(
    env: Any,
    agent_network: ModularAgentNetwork,
    replay_buffer: Optional[ModularReplayBuffer],
    obs_dim: Tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> BlackboardEngine:
    """
    Creates a standard Rainbow actor engine (BlackboardEngine).
    """
    obs_component = GymObservationComponent(env)
    step_component = GymStepComponent(env, obs_component)

    components = [
        obs_component,
        ResetNoiseComponent(agent_network),
        NetworkInferenceComponent(agent_network, obs_dim),
        ActionSelectorComponent(input_key="q_values", temperature=0.0),
        step_component,
        TelemetryComponent(name="rainbow_actor"),
    ]

    if replay_buffer is not None:
        components.append(BufferStoreComponent(replay_buffer))

    return BlackboardEngine(components=components, device=device)


def make_rainbow_learner(
    agent_network: ModularAgentNetwork,
    target_network: ModularAgentNetwork,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ModularReplayBuffer,
    gamma: float = 0.99,
    n_step: int = 3,
    clip_norm: float = 10.0,
    per_beta_schedule: Optional[Schedule] = None,
    device: torch.device = torch.device("cpu"),
) -> BlackboardEngine:
    """
    Creates a standard Rainbow learner (BlackboardEngine).
    """
    if per_beta_schedule is None:
        per_beta_schedule = ConstantSchedule(0.6)

    q_loss = QBootstrappingLoss(is_categorical=True, name="q_loss")
    priority_comp = LossPriorityComponent(loss_key="q_loss", reduction="max")
    buffer_update = PriorityUpdateComponent(priority_update_fn=replay_buffer.update_priorities)

    return BlackboardEngine(
        components=[
            ForwardPassComponent(agent_network, None),
            DistributionalTargetComponent(
                target_network=target_network,
                online_network=agent_network,
                gamma=gamma,
                n_step=n_step,
            ),
            q_loss,
            LossAggregatorComponent(loss_weights={"q_loss": 1.0}),
            priority_comp,
            OptimizerStepComponent(
                agent_network=agent_network,
                optimizers={"default": optimizer},
                max_grad_norm=clip_norm,
            ),
            buffer_update,
            BetaScheduleComponent(
                set_beta_fn=replay_buffer.set_beta,
                per_beta_schedule=per_beta_schedule,
            ),
            ResetNoiseComponent(agent_network=agent_network),
        ],
        device=device,
    )
