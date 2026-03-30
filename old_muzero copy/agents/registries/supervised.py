import torch
from typing import Any, Dict, List, Tuple
from old_muzero.agents.registries.base import register_agent
from old_muzero.agents.learner.losses import LossPipeline, ImitationLoss
from old_muzero.modules.utils import create_optimizer, get_lr_scheduler


def build_supervised_loss_pipeline(config, agent_network, device):
    # Extract representation from policy head
    pol_rep = agent_network.components["policy_head"].representation

    return LossPipeline(
        config,
        [
            ImitationLoss(
                device=device,
                representation=pol_rep,
                loss_fn=config.policy_loss_function,
                loss_factor=getattr(config, "policy_loss_factor", 1.0),
            )
        ],
    )


@register_agent("supervised")
def build_supervised(
    config: Any, agent_network: Any, device: torch.device
) -> Dict[str, Any]:
    # 1. Losses
    loss_pipeline = build_supervised_loss_pipeline(config, agent_network, device)

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
