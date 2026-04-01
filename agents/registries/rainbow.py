import torch
from typing import Any, Dict, List, Tuple, Optional
from agents.registries.base import register_agent
from agents.learner.losses import LossPipeline, QBootstrappingLoss
from agents.learner.losses.priorities import MaxLossPriorityComputer
from modules.utils import create_optimizer, get_lr_scheduler
from agents.learner.target_builders import (
    TemporalDifferenceBuilder,
    DistributionalTargetBuilder,
    TargetBuilderPipeline,
    SingleStepFormatter,
)
from agents.action_selectors.selectors import ArgmaxSelector


def build_rainbow_loss_pipeline(
    agent_network: Any,
    device: torch.device,
    minibatch_size: int,
    atom_size: int = 1,
    loss_function: Any = None,
    shape_validator: Optional[Any] = None,
):
    representation = None
    if (
        agent_network is not None
        and hasattr(agent_network, "components")
        and "q_head" in agent_network.components
    ):
        representation = agent_network.components["q_head"].representation

    is_distributional = atom_size > 1

    td_loss_module = QBootstrappingLoss(
        device=device,
        is_categorical=is_distributional,
        loss_fn=loss_function,
    )
    priority_computer = MaxLossPriorityComputer(loss_key="QBootstrappingLoss")

    return LossPipeline(
        modules=[td_loss_module],
        priority_computer=priority_computer,
        minibatch_size=minibatch_size,
        atom_size=atom_size,
        unroll_steps=0,  # Rainbow is single-step (TD)
        representations={td_loss_module.pred_key: representation}
        if representation
        else None,
        shape_validator=shape_validator,
    )


@register_agent("rainbow")
def build_rainbow(
    config: Any,
    agent_network: Any,
    device: torch.device,
    target_agent_network: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    # 1. Losses
    loss_pipeline = build_rainbow_loss_pipeline(
        agent_network=agent_network,
        device=device,
        minibatch_size=config.minibatch_size,
        atom_size=getattr(config, "atom_size", 1),
        loss_function=getattr(config, "loss_function", None),
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

    # 3. Callbacks
    from agents.learner.callbacks import TargetNetworkSyncCallback, ResetNoiseCallback

    callbacks = []
    if getattr(config, "use_noisy_net", False):
        callbacks.append(ResetNoiseCallback())

    if target_agent_network is not None:
        sync_interval = getattr(
            config,
            "transfer_interval",
            getattr(config, "target_network_update_freq", 100),
        )
        callbacks.append(
            TargetNetworkSyncCallback(
                target_network=target_agent_network,
                sync_interval=sync_interval,
                soft_update=getattr(config, "soft_update", False),
                ema_beta=getattr(config, "ema_beta", 0.99),
            )
        )

    # 4. Target Builder
    assert (
        target_agent_network is not None
    ), "Rainbow requires a target_agent_network for TD target building."

    is_distributional = getattr(config, "atom_size", 1) > 1
    builder_cls = (
        DistributionalTargetBuilder if is_distributional else TemporalDifferenceBuilder
    )

    target_builder = TargetBuilderPipeline(
        [
            builder_cls(
                target_network=target_agent_network,
                gamma=config.discount_factor,
                n_step=config.n_step,
                bootstrap_on_truncated=getattr(config, "bootstrap_on_truncated", False),
            ),
            SingleStepFormatter(),
        ]
    )

    return {
        "loss_pipeline": loss_pipeline,
        "optimizers": optimizers,
        "lr_schedulers": lr_schedulers,
        "target_builder": target_builder,
        "callbacks": callbacks,
        "observation_dtype": torch.uint8,
    }


def build_rainbow_network_components(
    config: Any, input_shape: Tuple[int, ...], num_actions: int, **kwargs
) -> Dict[str, Any]:
    from modules.backbones.factory import BackboneFactory
    from configs.modules.backbones.factory import BackboneConfigFactory
    from modules.heads.q import QHead, DuelingQHead
    from agents.learner.losses.representations import get_representation

    # 1. Feature Extractor
    # Check for 'backbone' with config_dict fallback.
    bb_cfg = getattr(config, "backbone", config.config_dict.get("backbone", {"type": "mlp"}))
    
    if isinstance(bb_cfg, dict):
        bb_cfg = BackboneConfigFactory.create(bb_cfg)
        
    backbone = BackboneFactory.create(bb_cfg, input_shape)
    feat_shape = backbone.output_shape

    # 2. Q Head
    representation = get_representation(config.head.output_strategy)

    neck = None
    if config.head.neck is not None:
        neck = BackboneFactory.create(config.head.neck, feat_shape)

    if getattr(config, "dueling", False):
        q_head = DuelingQHead(
            input_shape=feat_shape,
            num_actions=num_actions,
            representation=representation,
            value_hidden_widths=config.head.value_hidden_widths,
            advantage_hidden_widths=config.head.advantage_hidden_widths,
            neck=neck,
            noisy_sigma=config.arch.noisy_sigma,
            activation=config.arch.activation,
            norm_type=config.arch.norm_type,
        )
    else:
        q_head = QHead(
            input_shape=feat_shape,
            num_actions=num_actions,
            representation=representation,
            hidden_widths=config.head.hidden_widths,
            neck=neck,
            noisy_sigma=config.arch.noisy_sigma,
            activation=config.arch.activation,
            norm_type=config.arch.norm_type,
        )

    return {
        "components": {
            "feature_block": backbone,
            "q_head": q_head,
        },
        "metadata": {
            "minibatch_size": getattr(config, "minibatch_size", 1),
            "atom_size": getattr(config, "atom_size", 1),
            "support_range": getattr(config, "support_range", None),
        },
    }
