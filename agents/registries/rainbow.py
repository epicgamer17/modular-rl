import torch
from typing import Any, Dict, List, Tuple, Optional
from agents.registries.base import register_agent
from losses.losses import LossPipeline, StandardDQNLoss, C51Loss, ErrorPriority
from modules.utils import create_optimizer, get_lr_scheduler
from agents.learner.target_builders import (
    TemporalDifferenceBuilder,
)
from agents.action_selectors.selectors import ArgmaxSelector


def build_rainbow_loss_pipeline(config, agent_network, device):
    selector = ArgmaxSelector()
    representation = None
    if agent_network is not None and hasattr(agent_network, "components") and "q_head" in agent_network.components:
        representation = agent_network.components["q_head"].strategy.representation

    td_loss_module = (
        C51Loss(
            config=config,
            device=device,
            representation=representation,
            action_selector=selector,
        )
        if getattr(config, "atom_size", 1) > 1
        else StandardDQNLoss(
            config=config,
            device=device,
            representation=representation,
            action_selector=selector,
        )
    )
    priority_computer = ErrorPriority(prediction_key="q_values", target_key="returns", representation=representation)
    if getattr(config, "atom_size", 1) > 1:
        priority_computer = ErrorPriority(prediction_key="q_logits", target_key="values", representation=representation)
        
    return LossPipeline([td_loss_module], priority_computer=priority_computer)


@register_agent("rainbow")
def build_rainbow(
    config: Any,
    agent_network: Any,
    device: torch.device,
    target_agent_network: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    # 1. Losses
    loss_pipeline = build_rainbow_loss_pipeline(config, agent_network, device)

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
    # factory.py logic for target_builder
    assert (
        target_agent_network is not None
    ), "Rainbow requires a target_agent_network for TD target building."
    target_builder = TemporalDifferenceBuilder(
        target_network=target_agent_network,
        gamma=config.discount_factor,
        n_step=config.n_step,
        bootstrap_on_truncated=getattr(config, "bootstrap_on_truncated", False),
    )

    return {
        "loss_pipeline": loss_pipeline,
        "optimizers": optimizers,
        "lr_schedulers": lr_schedulers,
        "target_builder": target_builder,
        "callbacks": callbacks,
        "observation_dtype": torch.uint8,
    }
