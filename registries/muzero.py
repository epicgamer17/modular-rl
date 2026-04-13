import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List, Optional

from modules.agent_nets.modular import ModularAgentNetwork
from modules.embeddings.action_embedding import ActionEncoder
from modules.backbones.resnet import ResNetBackbone
from modules.backbones.conv import ConvBackbone
from modules.heads.value import ValueHead
from modules.heads.policy import PolicyHead
from modules.heads.reward import RewardHead
from modules.heads.to_play import ToPlayHead
from modules.world_models.muzero_world_model import MuzeroWorldModel
from modules.world_models.components.representation import Representation
from modules.world_models.components.dynamics import Dynamics

from search.backends.py_search.modular_search import ModularSearch
from actors.action_selectors.policy_sources import (
    BasePolicySource,
)
from actors.action_selectors.selectors import ActionSelector
from utils.schedule import StepwiseSchedule

from data.storage.circular import BufferConfig, ModularReplayBuffer
from data.processors import SequenceTensorProcessor, NStepUnrollProcessor
from data.writers import SharedCircularWriter
from data.samplers.prioritized import UniformSampler
from data.concurrency import TorchMPBackend

from core import BlackboardEngine
from components.neural import ForwardPassComponent
from components.losses import OptimizerStepComponent
from components.losses import LossAggregatorComponent
from components.memory import PriorityUpdateComponent
from components.losses import ExpectedValueErrorPriorityComponent
from components.losses import ValueLoss
from components.losses import PolicyLoss
from components.losses import RewardLoss
from components.losses import ToPlayLoss
from modules.representations import (
    ClassificationRepresentation,
    ScalarRepresentation,
    DiscreteSupportRepresentation,
)
from components.search import MCTSSearchComponent
from components.targets import (
    SequencePadderComponent,
    SequenceInfrastructureComponent,
    TwoHotProjectionComponent,
    ClassificationFormatterComponent,
    ScalarFormatterComponent,
    SequenceMaskComponent,
)
from components.losses import ShapeValidator
from components.environments import PettingZooObservationComponent, PettingZooStepComponent, GymObservationComponent, GymStepComponent
from components.selectors import (
    ActionSelectorComponent,
)
from components.memory import SequenceBufferComponent
from components.telemetry import TelemetryComponent
from components.learner_telemetry import MuzeroMultiplayerTelemetry


def make_muzero_network(
    obs_dim: Tuple[int, ...],
    num_actions: int,
    action_embedding_dim: int = 32,
    resnet_filters: List[int] = [24, 24, 24],
    unroll_steps: int = 5,
    device: torch.device = torch.device("cpu"),
) -> ModularAgentNetwork:
    """
    Creates a standard MuZero network.
    """
    action_encoder = ActionEncoder(num_actions, action_embedding_dim)

    representation = Representation(
        backbone=ResNetBackbone(
            input_shape=obs_dim,
            filters=resnet_filters,
            kernel_sizes=[3] * len(resnet_filters),
            strides=[1] * len(resnet_filters),
            norm_type="batch",
        )
    )
    hidden_state_shape = representation.output_shape

    dynamics = Dynamics(
        backbone=ResNetBackbone(
            input_shape=hidden_state_shape,
            filters=resnet_filters,
            kernel_sizes=[3] * len(resnet_filters),
            strides=[1] * len(resnet_filters),
            norm_type="batch",
        ),
        action_encoder=action_encoder,
        input_shape=hidden_state_shape,
        action_embedding_dim=action_embedding_dim,
    )

    reward_head = RewardHead(
        input_shape=hidden_state_shape,
        representation=ScalarRepresentation(),
        neck=ConvBackbone(
            input_shape=hidden_state_shape,
            filters=[16],
            kernel_sizes=[1],
            strides=[1],
            norm_type="batch",
        ),
    )

    to_play_head = ToPlayHead(
        input_shape=hidden_state_shape,
        num_players=2,
        neck=ConvBackbone(
            input_shape=hidden_state_shape,
            filters=[16],
            kernel_sizes=[1],
            strides=[1],
            norm_type="batch",
        ),
    )

    world_model = MuzeroWorldModel(
        representation=representation,
        dynamics=dynamics,
        reward_head=reward_head,
        to_play_head=to_play_head,
        num_actions=num_actions,
    )

    prediction_backbone = ResNetBackbone(
        input_shape=hidden_state_shape,
        filters=resnet_filters,
        kernel_sizes=[3] * len(resnet_filters),
        strides=[1] * len(resnet_filters),
        norm_type="batch",
    )

    value_head = ValueHead(
        input_shape=hidden_state_shape,
        representation=ScalarRepresentation(),
        neck=ConvBackbone(
            input_shape=hidden_state_shape,
            filters=[16],
            kernel_sizes=[1],
            strides=[1],
            norm_type="batch",
        ),
    )

    policy_head = PolicyHead(
        input_shape=hidden_state_shape,
        neck=ConvBackbone(
            input_shape=hidden_state_shape,
            filters=[16],
            kernel_sizes=[1],
            strides=[1],
            norm_type="batch",
        ),
        representation=ClassificationRepresentation(num_classes=num_actions),
    )

    agent_network = ModularAgentNetwork(
        components={
            "world_model": world_model,
            "prediction_backbone": prediction_backbone,
            "value_head": value_head,
            "policy_head": policy_head,
        },
        unroll_steps=unroll_steps,
        atom_size=1,
    ).to(device)

    return agent_network


def make_muzero_search_engine(
    num_actions: int,
    num_simulations: int = 25,
    discount_factor: float = 0.99,
    search_batch_size: int = 5,
    use_virtual_mean: bool = True,
    dirichlet_alpha: float = 0.3,
    dirichlet_fraction: float = 0.25,
    num_players: int = 2,
    device: torch.device = torch.device("cpu"),
) -> ModularSearch:
    """
    Creates a standard MuZero search engine (MCTS).
    """
    return ModularSearch(
        device=device,
        num_actions=num_actions,
        num_simulations=num_simulations,
        discount_factor=discount_factor,
        search_batch_size=search_batch_size,
        use_virtual_mean=use_virtual_mean,
        use_dirichlet=True,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_fraction=dirichlet_fraction,
        num_players=num_players,
    )


def make_muzero_replay_buffer(
    obs_dim: Tuple[int, ...],
    num_actions: int,
    buffer_size: int = 10000,
    batch_size: int = 8,
    unroll_steps: int = 5,
    td_steps: int = 10,
    discount_factor: float = 0.99,
    num_players: int = 2,
    player_map: Optional[Dict[str, int]] = None,
) -> ModularReplayBuffer:
    """
    Creates a standard MuZero replay buffer.
    """
    if player_map is None:
        player_map = {"player_1": 0, "player_2": 1}

    configs = [
        BufferConfig(
            "observations", shape=obs_dim, dtype=torch.float32, is_shared=True
        ),
        BufferConfig("actions", shape=(), dtype=torch.float16, is_shared=True),
        BufferConfig("rewards", shape=(), dtype=torch.float32, is_shared=True),
        BufferConfig("values", shape=(), dtype=torch.float32, is_shared=True),
        BufferConfig(
            "policies", shape=(num_actions,), dtype=torch.float32, is_shared=True
        ),
        BufferConfig("to_plays", shape=(), dtype=torch.int16, is_shared=True),
        BufferConfig("chances", shape=(1,), dtype=torch.int16, is_shared=True),
        BufferConfig("game_ids", shape=(), dtype=torch.int64, is_shared=True),
        BufferConfig("ids", shape=(), dtype=torch.int64, is_shared=True),
        BufferConfig("training_steps", shape=(), dtype=torch.int64, is_shared=True),
        BufferConfig("terminated", shape=(), dtype=torch.bool, is_shared=True),
        BufferConfig("truncated", shape=(), dtype=torch.bool, is_shared=True),
        BufferConfig("dones", shape=(), dtype=torch.bool, is_shared=True),
        BufferConfig(
            "legal_masks", shape=(num_actions,), dtype=torch.bool, is_shared=True
        ),
    ]

    input_processor = SequenceTensorProcessor(num_actions, num_players, player_map)
    output_processor = NStepUnrollProcessor(
        unroll_steps, td_steps, discount_factor, num_actions, num_players, buffer_size
    )

    return ModularReplayBuffer(
        max_size=buffer_size,
        batch_size=batch_size,
        buffer_configs=configs,
        input_processor=input_processor,
        output_processor=output_processor,
        writer=SharedCircularWriter(max_size=buffer_size),
        sampler=UniformSampler(),
        backend=TorchMPBackend(),
    )


def make_muzero_learner(
    agent_network: ModularAgentNetwork,
    replay_buffer: ModularReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    unroll_steps: int,
    num_actions: int,
    device: torch.device,
) -> BlackboardEngine:
    """
    Creates a standard MuZero learner (BlackboardEngine).
    """
    val_rep = agent_network.components["value_head"].representation
    pol_rep = agent_network.components["policy_head"].representation
    rew_rep = agent_network.components["world_model"].reward_head.representation
    tp_rep = agent_network.components["world_model"].to_play_head.representation

    shape_validator = ShapeValidator(
        minibatch_size=batch_size,
        unroll_steps=unroll_steps,
        num_actions=num_actions,
        atom_size=1,
    )
    priority_comp = ExpectedValueErrorPriorityComponent(value_representation=val_rep)
    buffer_update = PriorityUpdateComponent(priority_update_fn=replay_buffer.update_priorities)

    v_loss = ValueLoss(loss_fn=nn.functional.mse_loss, loss_factor=1.0)
    p_loss = PolicyLoss(loss_fn=nn.functional.cross_entropy, loss_factor=1.0)
    r_loss = RewardLoss(loss_fn=nn.functional.mse_loss, loss_factor=1.0)
    tp_loss = ToPlayLoss(loss_fn=nn.functional.cross_entropy, loss_factor=1.0)

    learner = BlackboardEngine(
        components=[
            ForwardPassComponent(agent_network, shape_validator),
            SequencePadderComponent(
                unroll_steps,
                keys=[
                    "data.values",
                    "data.rewards",
                    "data.policies",
                    "data.actions",
                    "data.to_plays",
                    "data.reward_mask",
                    "data.to_play_mask",
                    "data.policy_mask",
                    "data.dones",
                ],
            ),
            SequenceInfrastructureComponent(unroll_steps),
            SequenceMaskComponent(),
            TwoHotProjectionComponent(
                source_key="targets.values",
                dest_key="values",
                representation=val_rep,
            ) if isinstance(val_rep, DiscreteSupportRepresentation) else ScalarFormatterComponent(
                source_key="targets.values",
                dest_key="values",
                representation=val_rep,
            ),
            ClassificationFormatterComponent(
                source_key="targets.policies", dest_key="policies", representation=pol_rep
            ),
            TwoHotProjectionComponent(
                source_key="targets.rewards",
                dest_key="rewards",
                representation=rew_rep,
            ) if isinstance(rew_rep, DiscreteSupportRepresentation) else ScalarFormatterComponent(
                source_key="targets.rewards",
                dest_key="rewards",
                representation=rew_rep,
            ),
            ScalarFormatterComponent(
                source_key="targets.to_plays", dest_key="to_plays", representation=tp_rep
            ),
            v_loss,
            p_loss,
            r_loss,
            tp_loss,
            LossAggregatorComponent(
                loss_weights={
                    "value_loss": 1.0,
                    "reward_loss": 1.0,
                    "policy_loss": 1.0,
                    "consistency_loss": 1.0,
                    "to_play_loss": 1.0,
                }
            ),
            MuzeroMultiplayerTelemetry(),
            priority_comp,
            buffer_update,
            OptimizerStepComponent(
                agent_network=agent_network,
                optimizers={"default": optimizer},
            ),
        ],
        device=device,
    )

    return learner


def make_muzero_actor_engine(
    env: Any,
    agent_network: ModularAgentNetwork,
    search_engine: Any,
    replay_buffer: Optional[ModularReplayBuffer],
    obs_dim: Tuple[int, ...],
    num_actions: int,
    num_players: int,
    temperature_schedule: Optional[StepwiseSchedule] = None,
    exploration: bool = True,
    device: torch.device = torch.device("cpu"),
) -> BlackboardEngine:
    """
    Creates a standard MuZero actor engine (BlackboardEngine).
    """
    is_pz = hasattr(env, "possible_agents") or (hasattr(env, "unwrapped") and hasattr(env.unwrapped, "possible_agents"))

    if is_pz:
        obs_component = PettingZooObservationComponent(env)
        step_component = PettingZooStepComponent(env, obs_component)
    else:
        obs_component = GymObservationComponent(env)
        step_component = GymStepComponent(env, obs_component)

    components = [
        obs_component,
        MCTSSearchComponent(
            search_engine=search_engine,
            agent_network=agent_network,
        ),
    ]

    components.append(
        ActionSelectorComponent(
            input_key="probs",
            temperature=1.0 if exploration else 0.0,
            schedule=temperature_schedule,
            schedule_source="episode",
        )
    )
    
    # Step the environment AFTER action selection
    components.append(step_component)

    # Add TelemetryComponent to calculate stats like episode_length and scores
    components.append(TelemetryComponent(name="muzero_actor"))

    if replay_buffer is not None:
        components.append(SequenceBufferComponent(replay_buffer, num_players=num_players))

    return BlackboardEngine(components=components, device=device)
