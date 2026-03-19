import torch
from typing import Any, Dict, List, Tuple
from agents.registries.base import register_agent
from agents.learner.losses.losses import LossPipeline, PPOPolicyLoss, PPOValueLoss
from modules.utils import get_lr_scheduler
from agents.learner.callbacks import PPOEarlyStoppingCallback
from torch.optim.adam import Adam
from torch.optim.sgd import SGD


def build_ppo_loss_pipeline(config, agent_network, device):
    return LossPipeline(
        [
            PPOPolicyLoss(
                config=config,
                device=device,
                clip_param=config.clip_param,
                entropy_coefficient=config.entropy_coefficient,
                policy_strategy=getattr(
                    agent_network.components["policy_head"], "strategy", None
                ),
                optimizer_name="policy",
            ),
            PPOValueLoss(
                config=config,
                device=device,
                critic_coefficient=config.critic_coefficient,
                atom_size=getattr(config, "atom_size", 1),
                v_min=getattr(config, "v_min", None),
                v_max=getattr(config, "v_max", None),
                value_strategy=getattr(
                    agent_network.components["value_head"], "strategy", None
                ),
                optimizer_name="value",
            ),
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

    optimizers["policy"] = create_opt(
        agent_network.components["policy_head"].parameters(),
        getattr(config, "actor", config),
    )
    optimizers["value"] = create_opt(
        agent_network.components["value_head"].parameters(),
        getattr(config, "critic", config),
    )

    lr_schedulers["policy"] = get_lr_scheduler(optimizers["policy"], config)
    lr_schedulers["value"] = get_lr_scheduler(optimizers["value"], config)

    # 3. Callbacks
    callbacks = []
    if getattr(config, "use_early_stopping", False):
        callbacks.append(PPOEarlyStoppingCallback(config.early_stopping_kl))

    return {
        "loss_pipeline": loss_pipeline,
        "optimizers": optimizers,
        "lr_schedulers": lr_schedulers,
        "callbacks": callbacks,
    }
