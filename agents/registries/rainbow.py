import torch
from typing import Any, Dict, List, Tuple
from agents.registries.base import register_agent
from losses.losses import LossPipeline, StandardDQNLoss, C51Loss
from modules.utils import create_optimizer, get_lr_scheduler
from agents.learners.target_builders import DQNTargetBuilder
from agents.action_selectors.selectors import ArgmaxSelector

def build_rainbow_loss_pipeline(config, agent_network, device):
    selector = ArgmaxSelector()
    td_loss_module = (
        C51Loss(config=config, device=device, action_selector=selector)
        if getattr(config, "atom_size", 1) > 1
        else StandardDQNLoss(config=config, device=device, action_selector=selector)
    )
    return LossPipeline([td_loss_module])

@register_agent("rainbow")
def build_rainbow(config: Any, agent_network: Any, device: torch.device) -> Dict[str, Any]:
    # 1. Losses
    loss_pipeline = build_rainbow_loss_pipeline(config, agent_network, device)

    # 2. Optimizers
    opt = create_optimizer(agent_network.parameters(), config)
    optimizers = {"default": opt}
    lr_schedulers = {"default": get_lr_scheduler(opt, config)}

    # 3. Target Builder
    target_builder = DQNTargetBuilder(config, device)

    return {
        "loss_pipeline": loss_pipeline,
        "optimizers": optimizers,
        "lr_schedulers": lr_schedulers,
        "target_builder": target_builder,
        "observation_dtype": torch.uint8,
    }
