import torch
from typing import Any, Dict, List, Tuple
from torch import nn
from agents.registries.base import register_agent
from agents.learner.losses import LossPipeline, ClippedSurrogateLoss, ValueLoss
import torch.nn.functional as F
from modules.utils import get_lr_scheduler
from agents.learner.callbacks import MetricEarlyStopCallback
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from modules.backbones.factory import BackboneFactory
from modules.heads.value import ValueHead
from modules.heads.policy import PolicyHead
from agents.learner.losses.representations import get_representation
from configs.modules.backbones.factory import BackboneConfigFactory


def build_ppo_loss_pipeline(
    agent_network: Any,
    device: torch.device,
    clip_param: float,
    entropy_coefficient: float,
    critic_coefficient: float,
    minibatch_size: int,
    num_actions: int,
):
    # Extract representation from heads directly
    pol_rep = agent_network.components["policy_head"].representation
    val_rep = agent_network.components["value_head"].representation

    return LossPipeline(
        modules=[
            ClippedSurrogateLoss(
                device=device,
                representation=pol_rep,
                clip_param=clip_param,
                entropy_coefficient=entropy_coefficient,
                optimizer_name="policy",
            ),
            ValueLoss(
                device=device,
                representation=val_rep,
                target_key="returns",
                optimizer_name="value",
                loss_factor=critic_coefficient,
            ),
        ],
        minibatch_size=minibatch_size,
        num_actions=num_actions,
        unroll_steps=0,  # PPO is single-step
    )


@register_agent("ppo")
def build_ppo(config: Any, agent_network: Any, device: torch.device) -> Dict[str, Any]:
    # 1. Losses
    loss_pipeline = build_ppo_loss_pipeline(
        agent_network=agent_network,
        device=device,
        clip_param=config.clip_param,
        entropy_coefficient=config.entropy_coefficient,
        critic_coefficient=config.critic_coefficient,
        minibatch_size=config.minibatch_size,
        num_actions=config.game.num_actions,
    )

    # 2. Optimizers & Schedulers
    optimizers = {}
    lr_schedulers = {}

    def create_opt(params, sub_config_parent):
        # Determine which attribute to look at (actor/critic/default)
        opt_cls = getattr(sub_config_parent, "optimizer", Adam)

        if opt_cls == Adam:
            return Adam(
                params=params,
                lr=config.learning_rate,
                eps=getattr(config, "adam_epsilon", 1e-8),
                weight_decay=getattr(config, "weight_decay", 0.0),
            )
        elif opt_cls == SGD:
            return SGD(
                params=params,
                lr=config.learning_rate,
                momentum=getattr(config, "momentum", 0.0),
                weight_decay=getattr(config, "weight_decay", 0.0),
            )
        else:
            return opt_cls(params, lr=config.learning_rate)

    policy_params = list(agent_network.components["policy_head"].parameters())
    value_params = list(agent_network.components["value_head"].parameters())

    # If there is a shared backbone, add its parameters to the policy optimizer.
    # Note: In PPO with separate optimizers, representation learning typically happens
    # through the actor's update sequence. 
    if "backbone" in agent_network.components and not isinstance(
        agent_network.components["backbone"], nn.Identity
    ):
        policy_params += list(agent_network.components["backbone"].parameters())

    optimizers["policy"] = create_opt(
        policy_params,
        getattr(config, "actor", config),
    )
    optimizers["value"] = create_opt(
        value_params,
        getattr(config, "critic", config),
    )

    lr_schedulers["policy"] = get_lr_scheduler(optimizers["policy"], config)
    lr_schedulers["value"] = get_lr_scheduler(optimizers["value"], config)

    # 3. Callbacks
    callbacks = []
    if getattr(config, "use_early_stopping", False):
        callbacks.append(MetricEarlyStopCallback(threshold=config.early_stopping_kl))

    # 4. Target Builder
    from agents.learner.target_builders import (
        PassThroughTargetBuilder,
        TargetBuilderPipeline,
        SingleStepFormatter,
    )

    target_builder = TargetBuilderPipeline(
        [
            PassThroughTargetBuilder(
                ["values", "returns", "actions", "old_log_probs", "advantages"]
            ),
            SingleStepFormatter(),
        ]
    )

    return {
        "loss_pipeline": loss_pipeline,
        "optimizers": optimizers,
        "lr_schedulers": lr_schedulers,
        "callbacks": callbacks,
        "target_builder": target_builder,
    }


def build_ppo_network_components(
    config: Any, input_shape: Tuple[int, ...], num_actions: int, **kwargs
) -> Dict[str, Any]:
    # 1. Prediction Backbone
    # PPO typically uses a single backbone.
    # If neither 'prediction_backbone' nor 'backbone' is specified, we default to Identity (direct input to heads).
    bb_cfg = getattr(config, "prediction_backbone", None)
    if bb_cfg is None:
        bb_cfg = config.config_dict.get("backbone", None)

    if isinstance(bb_cfg, dict):
        bb_cfg = BackboneConfigFactory.create(bb_cfg)

    if bb_cfg:
        backbone = BackboneFactory.create(bb_cfg, input_shape)
        feat_shape = backbone.output_shape
    else:
        # Respect user's architectural intent: pass obs directly to head necks.
        backbone = nn.Identity()
        feat_shape = input_shape

    # 2. Prediction Heads
    val_rep = get_representation(config.value_head.output_strategy)
    val_neck = None
    if config.value_head.neck is not None:
        val_neck = BackboneFactory.create(config.value_head.neck, feat_shape)

    value_head = ValueHead(
        input_shape=feat_shape,
        representation=val_rep,
        neck=val_neck,
        noisy_sigma=config.arch.noisy_sigma,
    )

    pol_rep = get_representation(config.policy_head.output_strategy)
    pol_neck = None
    if config.policy_head.neck is not None:
        pol_neck = BackboneFactory.create(config.policy_head.neck, feat_shape)

    policy_head = PolicyHead(
        input_shape=feat_shape,
        representation=pol_rep,
        neck=pol_neck,
        noisy_sigma=config.arch.noisy_sigma,
    )

    return {
        "components": {
            "backbone": backbone,
            "value_head": value_head,
            "policy_head": policy_head,
        },
        "metadata": {
            "minibatch_size": getattr(config, "minibatch_size", 1),
        },
    }
