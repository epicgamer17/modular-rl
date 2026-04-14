import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List, Optional
import math

from modules.agent_nets.modular import ModularAgentNetwork
from modules.embeddings.action_embedding import ActionEncoder
from modules.backbones.resnet import ResNetBackbone
from modules.backbones.conv import ConvBackbone
from modules.heads.value import ValueHead
from modules.heads.policy import PolicyHead
from modules.heads.reward import ValuePrefixRewardHead
from modules.heads.to_play import ToPlayHead
from modules.world_models.muzero_world_model import MuzeroWorldModel
from modules.world_models.components.representation import Representation
from modules.world_models.components.dynamics import Dynamics
from modules.projectors.sim_siam import Projector

from search.backends.py_search.modular_search import ModularSearch
from utils.schedule import StepwiseSchedule

from data.storage.circular import ModularReplayBuffer

from core import BlackboardEngine
from components.neural import ForwardPassComponent
from components.losses import OptimizerStepComponent
from components.losses import LossAggregatorComponent
from components.memory import PriorityUpdateComponent
from components.losses import ExpectedValueErrorPriorityComponent
from components.losses import ValueLoss, PolicyLoss, RewardLoss, ToPlayLoss
from components.losses import ConsistencyLoss, LatentConsistencyComponent

from modules.representations import (
    ClassificationRepresentation,
    ScalarRepresentation,
    DiscreteSupportRepresentation,
)
from components.targets import (
    SequencePadderComponent,
    SequenceInfrastructureComponent,
    TwoHotProjectionComponent,
    ClassificationFormatterComponent,
    ScalarFormatterComponent,
    SequenceMaskComponent,
)
from components.losses import ShapeValidator
from components.learner_telemetry import MuzeroMultiplayerTelemetry

# Reuse standard components from muzero registry
from registries.muzero import (
    make_muzero_search_engine,
    make_muzero_replay_buffer,
    make_muzero_actor_engine,
)

make_efficient_zero_search_engine = make_muzero_search_engine
make_efficient_zero_replay_buffer = make_muzero_replay_buffer
make_efficient_zero_actor_engine = make_muzero_actor_engine

def make_efficient_zero_network(
    obs_dim: Tuple[int, ...],
    num_actions: int,
    action_embedding_dim: int = 32,
    resnet_filters: List[int] = [24, 24, 24],
    unroll_steps: int = 5,
    lstm_hidden_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> ModularAgentNetwork:
    """
    Creates an EfficientZero v1 network, characterized by a SimSiam projector
    and a ValuePrefixRewardHead. 
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

    reward_head = ValuePrefixRewardHead(
        input_shape=hidden_state_shape,
        representation=ScalarRepresentation(),
        lstm_hidden_size=lstm_hidden_size,
        lstm_horizon_len=unroll_steps,
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
    
    # SimSiam Projector
    flat_dim = math.prod(hidden_state_shape)
    projector = Projector(
        input_dim=flat_dim,
        projector_hidden_dim=256,
        projector_output_dim=256,
        predictor_hidden_dim=256,
        predictor_output_dim=256,
    )

    agent_network = ModularAgentNetwork(
        components={
            "world_model": world_model,
            "prediction_backbone": prediction_backbone,
            "value_head": value_head,
            "policy_head": policy_head,
            "projector": projector,
        },
        unroll_steps=unroll_steps,
        atom_size=1,
    ).to(device)

    return agent_network


def make_efficient_zero_learner(
    agent_network: ModularAgentNetwork,
    replay_buffer: ModularReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    unroll_steps: int,
    num_actions: int,
    device: torch.device,
) -> BlackboardEngine:
    """
    Creates an EfficientZero learner (BlackboardEngine) that includes the
    latent consistency matching component in the forward pipeline.
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
    
    # EfficientZero consistency loss
    c_loss = ConsistencyLoss(loss_factor=1.0) # Using "masks" as default

    learner = BlackboardEngine(
        components=[
            ForwardPassComponent(agent_network, shape_validator),
            # Extract real obs -> latents for projection targets
            LatentConsistencyComponent(agent_network),
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
            c_loss, # Add consistency loss explicitly
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
