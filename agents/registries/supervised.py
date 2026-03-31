import torch
from typing import Any, Dict, List, Tuple
from torch import nn
from agents.registries.base import register_agent
from agents.learner.losses import LossPipeline, ImitationLoss
from modules.utils import create_optimizer, get_lr_scheduler


def build_supervised_loss_pipeline(
    agent_network: Any,
    device: torch.device,
    minibatch_size: int,
    num_actions: int,
    policy_loss_function: Any,
    policy_loss_factor: float = 1.0,
):
    # Extract representation from policy head
    pol_rep = agent_network.components["policy_head"].representation

    return LossPipeline(
        modules=[
            ImitationLoss(
                device=device,
                representation=pol_rep,
                loss_fn=policy_loss_function,
                loss_factor=policy_loss_factor,
            )
        ],
        minibatch_size=minibatch_size,
        num_actions=num_actions,
        unroll_steps=0,  # SL is single-step
    )


@register_agent("supervised")
def build_supervised(
    config: Any, agent_network: Any, device: torch.device
) -> Dict[str, Any]:
    # 1. Losses
    loss_pipeline = build_supervised_loss_pipeline(
        agent_network=agent_network,
        device=device,
        minibatch_size=config.minibatch_size,
        num_actions=config.game.num_actions,
        policy_loss_function=config.policy_loss_function,
        policy_loss_factor=getattr(config, "policy_loss_factor", 1.0),
    )

    # 2. Setup Optimizers and Schedulers
    from torch.optim.adam import Adam
    from torch.optim.sgd import SGD

    optimizers = {}
    lr_schedulers = {}

    def create_opt(params, sub_config_parent):
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

    opt = create_opt(agent_network.parameters(), config)
    optimizers["default"] = opt
    lr_schedulers["default"] = get_lr_scheduler(opt, config)

    return {
        "loss_pipeline": loss_pipeline,
        "optimizers": optimizers,
        "lr_schedulers": lr_schedulers,
        "observation_dtype": torch.float32,
    }


def build_supervised_network_components(
    config: Any, input_shape: Tuple[int, ...], num_actions: int, **kwargs
) -> Dict[str, Any]:
    from modules.backbones.factory import BackboneFactory
    from configs.modules.backbones.factory import BackboneConfigFactory
    from modules.heads.policy import PolicyHead
    from agents.learner.losses.representations import get_representation

    # 1. Feature Extractor
    # Check for 'prediction_backbone' then 'backbone' with config_dict fallback.
    bb_cfg = getattr(config, "prediction_backbone", None)
    if bb_cfg is None:
        bb_cfg = getattr(config, "backbone", config.config_dict.get("backbone", None))
    
    if isinstance(bb_cfg, dict):
        bb_cfg = BackboneConfigFactory.create(bb_cfg)
    
    if bb_cfg:
        backbone = BackboneFactory.create(bb_cfg, input_shape)
        feat_shape = backbone.output_shape
    else:
        backbone = nn.Identity()
        feat_shape = input_shape

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
            "feature_block": backbone,
            "policy_head": policy_head,
        },
        "metadata": {
            "minibatch_size": getattr(config, "minibatch_size", 1),
        },
    }
