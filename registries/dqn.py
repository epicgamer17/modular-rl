import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List, Optional

from modules.agent_nets.modular import ModularAgentNetwork
from modules.backbones.mlp import MLPBackbone
from modules.heads.q import QHead
from modules.representations import ScalarRepresentation

from data.storage.circular import BufferConfig, ModularReplayBuffer, CircularWriter
from data.concurrency import LocalBackend
from data.processors import (
    StackedInputProcessor,
    StandardOutputProcessor,
    LegalMovesMaskProcessor,
)
from data.writers import SharedCircularWriter
from components.environments import GymObservationComponent, GymStepComponent
from components.telemetry import TelemetryComponent
from components.memory import BufferStoreComponent
from components.selectors import (
    NetworkInferenceComponent,
    EpsilonGreedySelectorComponent,
)

from core import BlackboardEngine
from components.neural import ForwardPassComponent
from components.losses import OptimizerStepComponent
from components.losses import LossAggregatorComponent
from components.losses import QBootstrappingLoss
from components.targets import TDTargetComponent


def make_dqn_network(
    obs_dim: Tuple[int, ...],
    num_actions: int,
    hidden_widths: List[int] = [64, 64],
    device: torch.device = torch.device("cpu"),
) -> ModularAgentNetwork:
    """
    Creates a standard DQN network.
    """
    representation = ScalarRepresentation()
    backbone = MLPBackbone(input_shape=obs_dim, widths=hidden_widths)
    head = QHead(
        input_shape=backbone.output_shape,
        num_actions=num_actions,
        representation=representation,
        hidden_widths=[],  # Backbone handles the depth
    )
    return ModularAgentNetwork(
        components={
            "feature_block": backbone,
            "q_head": head,
        },
        atom_size=1,
    ).to(device)


def make_dqn_replay_buffer(
    obs_dim: Tuple[int, ...],
    num_actions: int,
    max_size: int = 100000,
    batch_size: int = 32,
    shared: bool = False,
) -> ModularReplayBuffer:
    """
    Creates a standard DQN replay buffer.
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
            LegalMovesMaskProcessor(
                num_actions,
                input_key="next_legal_moves",
                output_key="next_legal_moves_masks",
            ),
        ]
    )

    writer_cls = SharedCircularWriter if shared else CircularWriter

    from data.samplers.prioritized import UniformSampler

    return ModularReplayBuffer(
        max_size=max_size,
        batch_size=batch_size,
        buffer_configs=buffer_configs,
        input_processor=input_processor,
        output_processor=StandardOutputProcessor(),
        writer=writer_cls(max_size=max_size),
        sampler=UniformSampler(),
        backend=LocalBackend(),
    )


def make_dqn_actor_engine(
    env: Any,
    agent_network: ModularAgentNetwork,
    replay_buffer: Optional[ModularReplayBuffer],
    obs_dim: Tuple[int, ...],
    epsilon: float = 0.05,
    device: torch.device = torch.device("cpu"),
) -> BlackboardEngine:
    """
    Creates a standard DQN actor engine (BlackboardEngine).
    """
    obs_component = GymObservationComponent(env)
    step_component = GymStepComponent(env, obs_component)

    components = [
        obs_component,
        NetworkInferenceComponent(agent_network, obs_dim),
        EpsilonGreedySelectorComponent(epsilon=epsilon),
        step_component,
        TelemetryComponent(name="dqn_actor"),
    ]

    if replay_buffer is not None:
        components.append(BufferStoreComponent(replay_buffer))

    return BlackboardEngine(components=components, device=device)


def make_dqn_learner(
    agent_network: ModularAgentNetwork,
    target_network: ModularAgentNetwork,
    optimizer: torch.optim.Optimizer,
    gamma: float = 0.99,
    clip_norm: float = 10.0,
    device: torch.device = torch.device("cpu"),
) -> BlackboardEngine:
    """
    Creates a standard DQN learner (BlackboardEngine).
    """
    q_loss = QBootstrappingLoss(is_categorical=False, name="q_loss")

    from core.contracts import (
        Key,
        Observation,
        Action,
        Reward,
        Done,
        Mask,
        SemanticType,
        LossScalar,
        Metric,
    )

    initial_keys = {
        Key("data.observations", Observation),
        Key("data.actions", Action),
        Key("data.rewards", Reward),
        Key("data.next_observations", Observation),
        Key("data.terminated", SemanticType),
        Key("data.truncated", SemanticType),
        Key("data.dones", Done),
        Key("data.next_legal_moves_masks", Mask),
    }
    target_keys = {Key("losses.total_loss", LossScalar)}

    return BlackboardEngine(
        components=[
            ForwardPassComponent(agent_network, None),
            TDTargetComponent(
                target_network=target_network,
                gamma=gamma,
            ),
            q_loss,
            LossAggregatorComponent(loss_weights={"q_loss": 1.0}),
            OptimizerStepComponent(
                agent_network=agent_network,
                optimizers={"default": optimizer},
                max_grad_norm=clip_norm,
            ),
        ],
        initial_keys=initial_keys,
        target_keys=target_keys,
        device=device,
    )
