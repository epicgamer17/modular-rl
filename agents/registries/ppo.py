import torch
from typing import Any, Dict, List, Tuple
from agents.registries.base import register_agent
from agents.learner.losses import LossPipeline, ClippedSurrogateLoss, ValueLoss
import torch.nn.functional as F
from modules.utils import get_lr_scheduler
from agents.learner.callbacks import MetricEarlyStopCallback
from torch.optim.adam import Adam
from torch.optim.sgd import SGD


def build_ppo_loss_pipeline(config, agent_network, device):
    # Extract representation from heads directly
    pol_rep = agent_network.components["behavior_heads"]["policy_logits"].representation
    val_rep = agent_network.components["behavior_heads"]["state_value"].representation

    from agents.learner.losses import ClippedValueLoss, ValueLoss
    
    # Select Value Loss module
    if getattr(config, "clip_value_loss", True):
        value_loss_module = ClippedValueLoss(
            device=device,
            representation=val_rep,
            clip_param=config.clip_param,
            target_key="returns",
            optimizer_name="default",
            loss_factor=config.critic_coefficient,
            name="value_loss",
        )
    else:
        value_loss_module = ValueLoss(
            device=device,
            representation=val_rep,
            target_key="returns",
            optimizer_name="default",
            loss_factor=config.critic_coefficient,
            name="value_loss",
        )

    return LossPipeline(
        config=config,
        modules=[
            ClippedSurrogateLoss(
                device=device,
                representation=pol_rep,
                clip_param=config.clip_param,
                entropy_coefficient=config.entropy_coefficient,
                optimizer_name="default",
                name="policy_loss",
            ),
            value_loss_module,
        ]
    )


@register_agent("ppo")
def build_ppo(config: Any, agent_network: Any, device: torch.device) -> Dict[str, Any]:
    # 1. Losses
    loss_pipeline = build_ppo_loss_pipeline(config, agent_network, device)

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

    # Standard PPO trains the entire network (backbone + heads) together
    optimizers["default"] = create_opt(
        agent_network.parameters(),
        config,
    )

    lr_schedulers["default"] = get_lr_scheduler(optimizers["default"], config)

    # 3. Callbacks
    callbacks = []
    if getattr(config, "use_early_stopping", False):
        callbacks.append(MetricEarlyStopCallback(threshold=config.early_stopping_kl))

    # 4. Target Builder
    from agents.learner.target_builders import (
        PassThroughTargetBuilder,
        SingleStepTargetPipeline,
    )

    math_builder = PassThroughTargetBuilder(
        ["values", "returns", "actions", "log_prob", "advantages"]
    )
    target_builder = SingleStepTargetPipeline([math_builder])

    return {
        "loss_pipeline": loss_pipeline,
        "optimizers": optimizers,
        "lr_schedulers": lr_schedulers,
        "callbacks": callbacks,
        "target_builder": target_builder,
    }
