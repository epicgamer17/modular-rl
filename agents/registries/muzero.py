import torch
from typing import Any, Dict, List, Tuple, Optional
from torch import nn
from agents.registries.base import register_agent
from agents.learner.losses import (
    LossPipeline,
    ValueLoss,
    PolicyLoss,
    RewardLoss,
    ToPlayLoss,
    ConsistencyLoss,
    ChanceQLoss,
    SigmaLoss,
    CommitmentLoss,
)
from agents.learner.callbacks import (
    ResetNoiseCallback,
    MetricEarlyStopCallback,
)
from modules.utils import get_lr_scheduler
from configs.modules.backbones.factory import BackboneConfigFactory
from modules.heads.value import ValueHead
from modules.heads.policy import PolicyHead
from modules.heads.to_play import ToPlayHead
from modules.heads.factory import HeadFactory
from modules.world_models.muzero_world_model import (
    MuzeroWorldModel,
    AfterstateDynamics,
    Dynamics,
    Representation,
)
from modules.projectors.sim_siam import Projector
from agents.learner.losses.representations import get_representation
from agents.learner.target_builders import (
    TargetBuilderPipeline,
    LatentConsistencyBuilder,
    MCTSExtractor,
    SequencePadder,
    SequenceMaskBuilder,
    SequenceInfrastructureBuilder,
    ChanceTargetBuilder,
)


from agents.learner.losses.representations import IdentityRepresentation, get_representation
from agents.learner.losses.priorities import ExpectedValueErrorPriorityComputer
from modules.world_models.muzero_world_model import MuzeroWorldModel
from modules.world_models.components.representation import Representation
from modules.world_models.components.dynamics import Dynamics, AfterstateDynamics
from modules.world_models.components.chance_encoder import ChanceEncoder
from modules.backbones.factory import BackboneFactory
from modules.heads.factory import HeadFactory
from modules.heads.to_play import ToPlayHead
from modules.projectors.sim_siam import Projector


def build_muzero_loss_pipeline(
    agent_network: Any,
    device: torch.device,
    minibatch_size: int,
    unroll_steps: int,
    num_actions: int,
    atom_size: int,
    value_loss_function: Any,
    value_loss_factor: float,
    policy_loss_function: Any,
    policy_loss_factor: float,
    reward_loss_function: Any,
    reward_loss_factor: float,
    num_players: int,
    to_play_loss_factor: float,
    consistency_loss_factor: float,
    stochastic: bool,
    chance_q_loss_factor: float = 0.0,
    sigma_loss_factor: float = 0.0,
):
    # Extract representations from heads
    val_rep = agent_network.components["value_head"].representation
    pol_rep = agent_network.components["policy_head"].representation
    rew_rep = agent_network.components["world_model"].reward_head.representation
    tp_rep = agent_network.components["world_model"].to_play_head.representation

    modules = [
        ValueLoss(
            device=device,
            representation=val_rep,
            loss_fn=value_loss_function,
            loss_factor=value_loss_factor,
        ),
        PolicyLoss(
            device=device,
            representation=pol_rep,
            loss_fn=policy_loss_function,
            loss_factor=policy_loss_factor,
        ),
        RewardLoss(
            device=device,
            representation=rew_rep,
            loss_fn=reward_loss_function,
            loss_factor=reward_loss_factor,
        ),
    ]

    if num_players > 1:
        modules.append(
            ToPlayLoss(
                device=device,
                representation=tp_rep,
                loss_factor=to_play_loss_factor,
            )
        )

    if consistency_loss_factor > 0:
        modules.append(
            ConsistencyLoss(
                device=device,
                representation=IdentityRepresentation(),
                agent_network=agent_network,
                loss_factor=consistency_loss_factor,
            )
        )

    if stochastic:
        as_val_rep = agent_network.components["afterstate_value_head"].representation
        sigma_rep = agent_network.components["world_model"].sigma_head.representation

        modules.extend(
            [
                ChanceQLoss(
                    device=device,
                    representation=as_val_rep,
                    loss_factor=chance_q_loss_factor,
                ),
                SigmaLoss(
                    device=device,
                    representation=sigma_rep,
                    loss_factor=sigma_loss_factor,
                ),
                CommitmentLoss(
                    device=device,
                    representation=IdentityRepresentation(),
                ),
            ]
        )

    priority_computer = ExpectedValueErrorPriorityComputer(value_representation=val_rep)
    return LossPipeline(
        modules=modules,
        priority_computer=priority_computer,
        minibatch_size=minibatch_size,
        unroll_steps=unroll_steps,
        num_actions=num_actions,
        atom_size=atom_size,
    )


@register_agent("muzero")
def build_muzero(
    config: Any, agent_network: Any, device: torch.device
) -> Dict[str, Any]:
    # 1. Losses
    loss_pipeline = build_muzero_loss_pipeline(
        agent_network=agent_network,
        device=device,
        minibatch_size=config.minibatch_size,
        unroll_steps=config.unroll_steps,
        num_actions=config.game.num_actions,
        atom_size=config.atom_size,
        value_loss_function=config.value_loss_function,
        value_loss_factor=config.value_loss_factor,
        policy_loss_function=config.policy_loss_function,
        policy_loss_factor=config.policy_loss_factor,
        reward_loss_function=config.reward_loss_function,
        reward_loss_factor=config.reward_loss_factor,
        num_players=config.game.num_players,
        to_play_loss_factor=config.to_play_loss_factor,
        consistency_loss_factor=config.consistency_loss_factor,
        stochastic=config.stochastic,
        chance_q_loss_factor=getattr(config, "chance_q_loss_factor", 0.0),
        sigma_loss_factor=getattr(config, "sigma_loss_factor", 0.0),
    )

    # 2. Setup Optimizers and Schedulers
    from torch.optim.adam import Adam
    from torch.optim.sgd import SGD

    optimizers = {}
    lr_schedulers = {}

    def create_opt(params, sub_config_parent):
        opt_cls = sub_config_parent.optimizer
        if opt_cls == Adam:
            return Adam(
                params=params,
                lr=config.learning_rate,
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay,
            )
        elif opt_cls == SGD:
            return SGD(
                params=params,
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        else:
            return opt_cls(params, lr=config.learning_rate)

    opt = create_opt(agent_network.parameters(), config)
    optimizers["default"] = opt
    lr_schedulers["default"] = get_lr_scheduler(opt, config)

    # 3. Callbacks
    callbacks = []
    if getattr(config, "use_noisy_net", False):
        callbacks.append(ResetNoiseCallback())
    if getattr(config, "use_early_stopping", False):
        callbacks.append(MetricEarlyStopCallback(threshold=config.early_stopping_kl))

    # 4. Target Builder
    builders = [
        MCTSExtractor(),
        SequencePadder(unroll_steps=config.unroll_steps),
        SequenceMaskBuilder(),
        SequenceInfrastructureBuilder(unroll_steps=config.unroll_steps),
    ]
    if getattr(config, "consistency_loss_factor", 0) > 0:
        builders.append(LatentConsistencyBuilder())

    if config.stochastic:
        builders.append(ChanceTargetBuilder())

    target_builder = TargetBuilderPipeline(builders)

    return {
        "loss_pipeline": loss_pipeline,
        "optimizers": optimizers,
        "lr_schedulers": lr_schedulers,
        "callbacks": callbacks,
        "target_builder": target_builder,
        "observation_dtype": torch.float32,
    }


def build_muzero_network_components(
    config: Any, input_shape: Tuple[int, ...], num_actions: int, **kwargs
) -> Dict[str, Any]:
    # 1. Representation Network
    rep_bb_cfg = getattr(config, "representation_backbone", config.config_dict.get("backbone", {"type": "mlp"}))
    if isinstance(rep_bb_cfg, dict):
        rep_bb_cfg = BackboneConfigFactory.create(rep_bb_cfg)
    representation_backbone = BackboneFactory.create(
        rep_bb_cfg, input_shape
    )
    representation = Representation(backbone=representation_backbone)

    hidden_state_shape = representation.output_shape

    # 2. Physics Engine (Dynamics & Heads)
    stochastic = getattr(config, "stochastic", False)
    num_chance = getattr(config, "num_chance", 0)
    action_embedding_dim = getattr(config, "action_embedding_dim", 0)

    # Action Encoder for Dynamics
    from modules.embeddings.action_embedding import ActionEncoder

    def create_action_encoder(n_actions, is_dyn=True):
        return ActionEncoder(
            action_space_size=n_actions,
            embedding_dim=action_embedding_dim,
            is_continuous=not config.game.is_discrete,
            single_action_plane=is_dyn,
        )

    afterstate_dynamics = None
    shared_backbone = None
    sigma_head = None
    encoder = None

    if stochastic:
        as_bb_cfg = getattr(config, "afterstate_dynamics_backbone", config.config_dict.get("backbone", {"type": "mlp"}))
        if isinstance(as_bb_cfg, dict):
            as_bb_cfg = BackboneConfigFactory.create(as_bb_cfg)
        afterstate_dynamics_backbone = BackboneFactory.create(
            as_bb_cfg, hidden_state_shape
        )
        as_encoder = create_action_encoder(num_actions, is_dyn=False)
        afterstate_dynamics = AfterstateDynamics(
            backbone=afterstate_dynamics_backbone,
            action_encoder=as_encoder,
            input_shape=hidden_state_shape,
            action_embedding_dim=action_embedding_dim,
        )

        dyn_bb_cfg = getattr(config, "dynamics_backbone", config.config_dict.get("backbone", {"type": "mlp"}))
        if isinstance(dyn_bb_cfg, dict):
            dyn_bb_cfg = BackboneConfigFactory.create(dyn_bb_cfg)
        dynamics_backbone = BackboneFactory.create(
            dyn_bb_cfg, hidden_state_shape
        )
        dyn_encoder = create_action_encoder(num_chance, is_dyn=True)
        dynamics = Dynamics(
            backbone=dynamics_backbone,
            action_encoder=dyn_encoder,
            input_shape=hidden_state_shape,
            action_embedding_dim=action_embedding_dim,
        )

        pred_bb_cfg = getattr(config, "prediction_backbone", config.config_dict.get("backbone", {"type": "mlp"}))
        if isinstance(pred_bb_cfg, dict):
            pred_bb_cfg = BackboneConfigFactory.create(pred_bb_cfg)
        shared_backbone = BackboneFactory.create(
            pred_bb_cfg, hidden_state_shape
        )
        sigma_head = HeadFactory.create(
            config.chance_probability_head,
            config.arch,
            input_shape=shared_backbone.output_shape,
            num_chance_codes=num_chance,
        )

        encoder_input_shape = list(input_shape)
        encoder_input_shape[0] *= 2
        encoder = ChanceEncoder(
            config, tuple(encoder_input_shape), num_codes=num_chance
        )
    else:
        dyn_bb_cfg = getattr(config, "dynamics_backbone", config.config_dict.get("backbone", {"type": "mlp"}))
        if isinstance(dyn_bb_cfg, dict):
            dyn_bb_cfg = BackboneConfigFactory.create(dyn_bb_cfg)
        dynamics_backbone = BackboneFactory.create(
            dyn_bb_cfg, hidden_state_shape
        )
        dyn_encoder = create_action_encoder(num_actions, is_dyn=True)
        dynamics = Dynamics(
            backbone=dynamics_backbone,
            action_encoder=dyn_encoder,
            input_shape=hidden_state_shape,
            action_embedding_dim=action_embedding_dim,
        )

    # Physics Heads
    r_rep = get_representation(config.reward_head.output_strategy)
    reward_head = HeadFactory.create(
        config.reward_head,
        config.arch,
        input_shape=dynamics.output_shape,
        representation=r_rep,
    )

    tp_rep = get_representation(config.to_play_head.output_strategy)
    tp_neck = None
    if config.to_play_head.neck is not None:
        tp_neck = BackboneFactory.create(config.to_play_head.neck, dynamics.output_shape)

    to_play_head = ToPlayHead(
        input_shape=dynamics.output_shape,
        num_players=config.game.num_players,
        neck=tp_neck,
        noisy_sigma=config.arch.noisy_sigma,
        representation=tp_rep,
    )

    world_model_cls = kwargs.get("world_model_cls", MuzeroWorldModel)
    world_model = world_model_cls(
        representation=representation,
        dynamics=dynamics,
        reward_head=reward_head,
        to_play_head=to_play_head,
        num_actions=num_actions,
        stochastic=stochastic,
        afterstate_dynamics=afterstate_dynamics,
        sigma_head=sigma_head,
        encoder=encoder,
        shared_backbone=shared_backbone,
        num_chance=num_chance,
        use_true_chance_codes=getattr(config, "use_true_chance_codes", False),
        consistency_loss_factor=getattr(config, "consistency_loss_factor", 0.0),
    )

    # 3. Prediction Backbone and Heads
    pred_bb_cfg = getattr(config, "prediction_backbone", config.config_dict.get("backbone", {"type": "mlp"}))
    if isinstance(pred_bb_cfg, dict):
        pred_bb_cfg = BackboneConfigFactory.create(pred_bb_cfg)
    prediction_backbone = BackboneFactory.create(
        pred_bb_cfg, hidden_state_shape
    )
    prediction_feat_shape = prediction_backbone.output_shape

    val_rep = get_representation(config.value_head.output_strategy)
    val_neck = None
    if config.value_head.neck is not None:
        val_neck = BackboneFactory.create(config.value_head.neck, prediction_feat_shape)

    value_head = ValueHead(
        input_shape=prediction_feat_shape,
        representation=val_rep,
        neck=val_neck,
        noisy_sigma=config.arch.noisy_sigma,
    )

    pol_rep = get_representation(config.policy_head.output_strategy)
    pol_neck = None
    if config.policy_head.neck is not None:
        pol_neck = BackboneFactory.create(config.policy_head.neck, prediction_feat_shape)

    policy_head = PolicyHead(
        input_shape=prediction_feat_shape,
        representation=pol_rep,
        neck=pol_neck,
        noisy_sigma=config.arch.noisy_sigma,
    )

    components = {
        "world_model": world_model,
        "prediction_backbone": prediction_backbone,
        "value_head": value_head,
        "policy_head": policy_head,
    }

    if stochastic:
        as_val_rep = get_representation(config.afterstate_value_head.output_strategy)
        as_val_neck = None
        if config.afterstate_value_head.neck is not None:
            as_val_neck = BackboneFactory.create(config.afterstate_value_head.neck, shared_backbone.output_shape)

        components["afterstate_value_head"] = ValueHead(
            input_shape=shared_backbone.output_shape,
            representation=as_val_rep,
            neck=as_val_neck,
            noisy_sigma=config.arch.noisy_sigma,
        )

    # 4. Auxiliary Component: SIM-SIAM Projector (EfficientZero)
    if getattr(config, "consistency_loss_factor", 0) > 0:
        components["projector"] = Projector(
            input_dim=flat_hidden_dim,
            projector_hidden_dim=config.projector_hidden_dim,
            projector_output_dim=config.projector_output_dim,
            predictor_hidden_dim=config.predictor_hidden_dim,
            predictor_output_dim=config.predictor_output_dim,
        )

    return {
        "components": components,
        "metadata": {
            "stochastic": stochastic,
            "unroll_steps": getattr(config, "unroll_steps", 0),
            "minibatch_size": getattr(config, "minibatch_size", 1),
            "atom_size": getattr(config, "atom_size", 1),
            "support_range": getattr(config, "support_range", None),
        },
    }
