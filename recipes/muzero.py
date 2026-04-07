"""MuZero component registry.

Assembles the full MuZero stack: world-model agent network, MCTS search,
sequence replay buffer, and the multi-head loss pipeline.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from modules.agent_nets.modular import ModularAgentNetwork
from modules.backbones.conv import ConvBackbone
from modules.backbones.resnet import ResNetBackbone
from modules.embeddings.action_embedding import ActionEncoder
from modules.heads.policy import PolicyHead
from modules.heads.reward import RewardHead
from modules.heads.to_play import ToPlayHead
from modules.heads.value import ValueHead
from modules.world_models.components.dynamics import Dynamics
from modules.world_models.components.representation import Representation
from modules.world_models.muzero_world_model import MuzeroWorldModel

from actors.action_selectors.decorators import TemperatureSelector
from actors.action_selectors.selectors import CategoricalSelector
from actors.action_selectors.policy_sources import SearchPolicySource
from search.backends.py_search.modular_search import ModularSearch

from data.concurrency import TorchMPBackend
from data.processors import NStepUnrollProcessor, SequenceTensorProcessor
from data.samplers.prioritized import UniformSampler
from data.storage.circular import BufferConfig, ModularReplayBuffer
from data.writers import SharedCircularWriter

from learner.base import UniversalLearner
from learner.losses.loss_pipeline import LossPipeline
from learner.losses.policy import PolicyLoss
from learner.losses.priorities import ExpectedValueErrorPriorityComputer
from learner.losses.representations import (
    ClassificationRepresentation,
    ScalarRepresentation,
)
from learner.losses.reward import RewardLoss
from learner.losses.shape_validator import ShapeValidator
from learner.losses.to_play import ToPlayLoss
from learner.losses.value import ValueLoss
from learner.pipeline.targets import (
    MCTSExtractor,
    SequenceInfrastructureBuilder,
    SequenceMaskBuilder,
    SequencePadder,
    TargetBuilderPipeline,
    TargetFormatter,
)


def build_muzero_network(
    obs_dim: Tuple[int, ...],
    num_actions: int,
    hidden_filters: int = 24,
    neck_filters: int = 16,
    num_res_blocks: int = 3,
    action_embedding_dim: int = 32,
    num_players: int = 2,
    unroll_steps: int = 5,
    device: torch.device = torch.device("cpu"),
) -> ModularAgentNetwork:
    """Build a MuZero agent network with ResNet backbones."""
    action_encoder = ActionEncoder(num_actions, action_embedding_dim)

    representation = Representation(
        backbone=ResNetBackbone(
            input_shape=obs_dim,
            filters=[hidden_filters] * num_res_blocks,
            kernel_sizes=[3] * num_res_blocks,
            strides=[1] * num_res_blocks,
            norm_type="batch",
        )
    )
    hidden_state_shape = representation.output_shape

    dynamics = Dynamics(
        backbone=ResNetBackbone(
            input_shape=hidden_state_shape,
            filters=[hidden_filters] * num_res_blocks,
            kernel_sizes=[3] * num_res_blocks,
            strides=[1] * num_res_blocks,
            norm_type="batch",
        ),
        action_encoder=action_encoder,
        input_shape=hidden_state_shape,
        action_embedding_dim=action_embedding_dim,
    )

    def _make_neck(shape: Tuple[int, ...]) -> ConvBackbone:
        return ConvBackbone(
            input_shape=shape,
            filters=[neck_filters],
            kernel_sizes=[1],
            strides=[1],
            norm_type="batch",
        )

    world_model = MuzeroWorldModel(
        representation=representation,
        dynamics=dynamics,
        reward_head=RewardHead(
            input_shape=hidden_state_shape,
            representation=ScalarRepresentation(),
            neck=_make_neck(hidden_state_shape),
        ),
        to_play_head=ToPlayHead(
            input_shape=hidden_state_shape,
            num_players=num_players,
            neck=_make_neck(hidden_state_shape),
        ),
        num_actions=num_actions,
    )

    prediction_backbone = ResNetBackbone(
        input_shape=hidden_state_shape,
        filters=[hidden_filters] * num_res_blocks,
        kernel_sizes=[3] * num_res_blocks,
        strides=[1] * num_res_blocks,
        norm_type="batch",
    )

    value_head = ValueHead(
        input_shape=hidden_state_shape,
        representation=ScalarRepresentation(),
        neck=_make_neck(hidden_state_shape),
    )

    policy_head = PolicyHead(
        input_shape=hidden_state_shape,
        neck=_make_neck(hidden_state_shape),
        representation=ClassificationRepresentation(num_classes=num_actions),
    )

    return ModularAgentNetwork(
        components={
            "world_model": world_model,
            "prediction_backbone": prediction_backbone,
            "value_head": value_head,
            "policy_head": policy_head,
        },
        unroll_steps=unroll_steps,
        atom_size=1,
    ).to(device)


def build_muzero_buffer(
    obs_dim: Tuple[int, ...],
    num_actions: int,
    num_players: int,
    player_map: Dict[str, int],
    buffer_size: int,
    batch_size: int,
    unroll_steps: int,
    td_steps: int,
    discount_factor: float,
) -> ModularReplayBuffer:
    """Build a MuZero replay buffer with sequence processing."""
    configs = [
        BufferConfig("observations", shape=obs_dim, dtype=torch.float32, is_shared=True),
        BufferConfig("actions", shape=(), dtype=torch.float16, is_shared=True),
        BufferConfig("rewards", shape=(), dtype=torch.float32, is_shared=True),
        BufferConfig("values", shape=(), dtype=torch.float32, is_shared=True),
        BufferConfig("policies", shape=(num_actions,), dtype=torch.float32, is_shared=True),
        BufferConfig("to_plays", shape=(), dtype=torch.int16, is_shared=True),
        BufferConfig("chances", shape=(1,), dtype=torch.int16, is_shared=True),
        BufferConfig("game_ids", shape=(), dtype=torch.int64, is_shared=True),
        BufferConfig("ids", shape=(), dtype=torch.int64, is_shared=True),
        BufferConfig("training_steps", shape=(), dtype=torch.int64, is_shared=True),
        BufferConfig("terminated", shape=(), dtype=torch.bool, is_shared=True),
        BufferConfig("truncated", shape=(), dtype=torch.bool, is_shared=True),
        BufferConfig("dones", shape=(), dtype=torch.bool, is_shared=True),
        BufferConfig("legal_masks", shape=(num_actions,), dtype=torch.bool, is_shared=True),
    ]

    return ModularReplayBuffer(
        max_size=buffer_size,
        batch_size=batch_size,
        buffer_configs=configs,
        input_processor=SequenceTensorProcessor(num_actions, num_players, player_map),
        output_processor=NStepUnrollProcessor(
            unroll_steps, td_steps, discount_factor, num_actions, num_players, buffer_size
        ),
        writer=SharedCircularWriter(max_size=buffer_size),
        sampler=UniformSampler(),
        backend=TorchMPBackend(),
    )


def build_muzero_learner(
    agent_network: ModularAgentNetwork,
    obs_dim: Tuple[int, ...],
    num_actions: int,
    batch_size: int,
    unroll_steps: int,
    learning_rate: float = 1e-3,
    clipnorm: float = 5.0,
    device: torch.device = torch.device("cpu"),
) -> UniversalLearner:
    """Build the MuZero learner with multi-head loss pipeline."""
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

    loss_pipeline = LossPipeline(
        modules=[
            ValueLoss(device=device, loss_fn=nn.functional.mse_loss, loss_factor=1.0),
            PolicyLoss(device=device, loss_fn=nn.functional.cross_entropy, loss_factor=1.0),
            RewardLoss(device=device, loss_fn=nn.functional.mse_loss, loss_factor=1.0),
            ToPlayLoss(device=device, loss_factor=1.0),
        ],
        priority_computer=ExpectedValueErrorPriorityComputer(value_representation=val_rep),
        minibatch_size=batch_size,
        unroll_steps=unroll_steps,
        num_actions=num_actions,
        atom_size=1,
        representations={
            "values": val_rep,
            "policies": pol_rep,
            "rewards": rew_rep,
            "to_plays": tp_rep,
        },
        shape_validator=shape_validator,
    )

    target_builder = TargetBuilderPipeline(
        builders=[
            MCTSExtractor(),
            SequencePadder(unroll_steps),
            SequenceMaskBuilder(),
            SequenceInfrastructureBuilder(unroll_steps),
            TargetFormatter(
                {"values": val_rep, "policies": pol_rep, "rewards": rew_rep, "to_plays": tp_rep}
            ),
        ]
    )

    optimizer = {
        "default": torch.optim.Adam(
            agent_network.parameters(), lr=learning_rate, eps=1e-5
        )
    }

    return UniversalLearner(
        agent_network=agent_network,
        device=device,
        optimizer=optimizer,
        loss_pipeline=loss_pipeline,
        target_builder=target_builder,
        num_actions=num_actions,
        observation_dimensions=obs_dim,
        observation_dtype=torch.float32,
        shape_validator=shape_validator,
        clipnorm=clipnorm,
    )


def build_muzero_search(
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
    """Build a MuZero MCTS search engine."""
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


def build_muzero_components(
    obs_dim: Tuple[int, ...],
    num_actions: int,
    num_players: int = 2,
    player_map: Optional[Dict[str, int]] = None,
    buffer_size: int = 10000,
    batch_size: int = 8,
    unroll_steps: int = 5,
    td_steps: int = 10,
    discount_factor: float = 0.99,
    learning_rate: float = 1e-3,
    num_simulations: int = 25,
    search_batch_size: int = 5,
    dirichlet_alpha: float = 0.3,
    dirichlet_fraction: float = 0.25,
    temperature_steps: Optional[list] = None,
    temperature_values: Optional[list] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Build all MuZero components and return them as a dictionary.

    Returns:
        Dict with keys: agent_network, search_engine, action_selector,
        replay_buffer, learner, policy_source.
    """
    if player_map is None:
        player_map = {f"player_{i}": i for i in range(num_players)}
    if temperature_steps is None:
        temperature_steps = [5, 10]
    if temperature_values is None:
        temperature_values = [1.0, 0.5, 0.0]

    from utils.schedule import StepwiseSchedule

    agent_network = build_muzero_network(
        obs_dim=obs_dim,
        num_actions=num_actions,
        num_players=num_players,
        unroll_steps=unroll_steps,
        device=device,
    )

    search_engine = build_muzero_search(
        num_actions=num_actions,
        num_simulations=num_simulations,
        discount_factor=discount_factor,
        search_batch_size=search_batch_size,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_fraction=dirichlet_fraction,
        num_players=num_players,
        device=device,
    )

    inner_selector = CategoricalSelector(exploration=True)
    action_selector = TemperatureSelector(
        inner_selector=inner_selector,
        schedule=StepwiseSchedule(steps=temperature_steps, values=temperature_values),
    )

    replay_buffer = build_muzero_buffer(
        obs_dim=obs_dim,
        num_actions=num_actions,
        num_players=num_players,
        player_map=player_map,
        buffer_size=buffer_size,
        batch_size=batch_size,
        unroll_steps=unroll_steps,
        td_steps=td_steps,
        discount_factor=discount_factor,
    )

    learner = build_muzero_learner(
        agent_network=agent_network,
        obs_dim=obs_dim,
        num_actions=num_actions,
        batch_size=batch_size,
        unroll_steps=unroll_steps,
        learning_rate=learning_rate,
        device=device,
    )

    policy_source = SearchPolicySource(
        search_engine=search_engine,
        agent_network=agent_network,
        input_shape=obs_dim,
        num_actions=num_actions,
    )

    return {
        "agent_network": agent_network,
        "search_engine": search_engine,
        "action_selector": action_selector,
        "replay_buffer": replay_buffer,
        "learner": learner,
        "policy_source": policy_source,
    }
