import torch
from typing import Any, Dict, List, Tuple, Optional, Callable
from agents.registries.base import register_agent
from agents.learner.losses import LossPipeline, QBootstrappingLoss
from agents.learner.losses.priorities import MaxLossPriorityComputer
from modules.utils import create_optimizer, get_lr_scheduler
from agents.learner.target_builders import (
    TemporalDifferenceBuilder,
    DistributionalTargetBuilder,
    SingleStepTargetPipeline,
)
from agents.action_selectors.selectors import ArgmaxSelector


def build_rainbow_loss_pipeline(config, agent_network, device):
    representation = None
    if (
        agent_network is not None
        and hasattr(agent_network, "components")
        and "behavior_heads" in agent_network.components
        and "q_logits" in agent_network.components["behavior_heads"]
    ):
        representation = agent_network.components["behavior_heads"][
            "q_logits"
        ].representation

    is_distributional = getattr(config, "atom_size", 1) > 1

    td_loss_module = QBootstrappingLoss(
        device=device,
        representation=representation,
        is_categorical=is_distributional,
        loss_fn=getattr(config, "loss_function", None),
    )

    # --- FIXED PRIORITY COMPUTER LOGIC ---
    # if is_distributional:
    #     from agents.learner.losses.priorities import ExpectedValueErrorPriorityComputer

    #     priority_computer = ExpectedValueErrorPriorityComputer(
    #         value_representation=representation,
    #         target_key="q_logits",  # Ensure this matches your DistributionalTargetBuilder output key
    #         pred_key="q_logits",
    #     )
    # else:
    priority_computer = MaxLossPriorityComputer(loss_key="QBootstrappingLoss")

    return LossPipeline(config, [td_loss_module], priority_computer=priority_computer)


@register_agent("rainbow")
def build_rainbow(
    config: Any,
    agent_network: Any,
    device: torch.device,
    target_agent_network: torch.nn.Module,
    priority_update_fn: Optional[Callable] = None,
    set_beta_fn: Optional[Callable] = None,
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
    from agents.learner.callbacks import (
        TargetNetworkSyncCallback,
        ResetNoiseCallback,
        PriorityUpdaterCallback,
    )
    from utils.schedule import create_schedule

    callbacks = []
    if getattr(config, "use_noisy_net", False):
        callbacks.append(ResetNoiseCallback(target_agent_network))

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

    # 4. Priority Update Callback (PER)
    if getattr(config, "use_per", True) and priority_update_fn is not None:
        callbacks.append(
            PriorityUpdaterCallback(
                priority_update_fn=priority_update_fn,
                set_beta_fn=set_beta_fn,
                per_beta_schedule=create_schedule(config.per_beta_schedule),
            )
        )

    # 5. Target Builder
    assert (
        target_agent_network is not None
    ), "Rainbow requires a target_agent_network for TD target building."

    is_distributional = getattr(config, "atom_size", 1) > 1
    builder_cls = (
        DistributionalTargetBuilder if is_distributional else TemporalDifferenceBuilder
    )

    math_builder = builder_cls(
        target_network=target_agent_network,
        gamma=config.discount_factor,
        n_step=config.n_step,
        bootstrap_on_truncated=getattr(config, "bootstrap_on_truncated", False),
    )

    # Automatically applies SingleStepFormatter
    target_builder = SingleStepTargetPipeline([math_builder])

    return {
        "loss_pipeline": loss_pipeline,
        "optimizers": optimizers,
        "lr_schedulers": lr_schedulers,
        "target_builder": target_builder,
        "callbacks": callbacks,
        "observation_dtype": torch.float32,
    }
