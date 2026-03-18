import torch
from typing import Any, Dict, List, Tuple
from agents.registries.base import register_agent
from losses.losses import LossPipeline, ImitationLoss
from modules.utils import create_optimizer, get_lr_scheduler


def build_supervised_loss_pipeline(config, agent_network, device):
    return LossPipeline(
        [
            ImitationLoss(
                config,
                device,
                agent_network.num_actions,
                representation=agent_network.components[
                    "policy_head"
                ].strategy.representation,
            )
        ]
    )


@register_agent("supervised")
def build_supervised(
    config: Any, agent_network: Any, device: torch.device
) -> Dict[str, Any]:
    # 1. Losses
    loss_pipeline = build_supervised_loss_pipeline(config, agent_network, device)

    # 2. Optimizers
    opt = create_optimizer(agent_network.parameters(), config)
    optimizers = {"default": opt}
    lr_schedulers = {"default": get_lr_scheduler(opt, config)}

    return {
        "loss_pipeline": loss_pipeline,
        "optimizers": optimizers,
        "lr_schedulers": lr_schedulers,
        "observation_dtype": torch.float32,
    }
