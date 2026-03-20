import torch
from typing import Any, Dict, List, Tuple
from agents.registries.base import register_agent
from agents.learner.losses import (
    LossPipeline,
    ValueLoss,
    PolicyLoss,
    RewardLoss,
    ToPlayLoss,
    ConsistencyLoss,
    ChanceQLoss,
    SigmaLoss,
    CommitmentLoss,
)
from agents.learner.callbacks import (
    ResetNoiseCallback,
)
from modules.utils import get_lr_scheduler
from agents.learner.target_builders import (
    TargetBuilderPipeline,
    TrajectoryGradientScaleBuilder,
    LatentConsistencyBuilder,
    MuZeroTargetBuilder,
)


from agents.learner.losses.representations import IdentityRepresentation
from agents.learner.losses.priorities import ExpectedValueErrorPriorityComputer


def build_muzero_loss_pipeline(config, agent_network, device):
    # Extract representations from heads
    val_rep = agent_network.components["value_head"].representation
    pol_rep = agent_network.components["policy_head"].representation
    rew_rep = agent_network.components["world_model"].reward_head.representation
    tp_rep = agent_network.components["world_model"].to_play_head.representation

    modules = [
        ValueLoss(config, device, representation=val_rep),
        PolicyLoss(config, device, representation=pol_rep),
        RewardLoss(config, device, representation=rew_rep),
    ]
    if config.game.num_players > 1:
        modules.append(ToPlayLoss(config, device, representation=tp_rep))
    if config.consistency_loss_factor > 0:
        modules.append(
            ConsistencyLoss(
                config,
                device,
                representation=IdentityRepresentation(),
                agent_network=agent_network,
            )
        )
    if config.stochastic:
        as_val_rep = agent_network.components["afterstate_value_head"].representation
        sigma_rep = agent_network.components["world_model"].sigma_head.representation

        modules.extend(
            [
                ChanceQLoss(config, device, representation=as_val_rep),
                SigmaLoss(config, device, representation=sigma_rep),
                CommitmentLoss(
                    config, device, representation=IdentityRepresentation()
                ),
            ]
        )
    priority_computer = ExpectedValueErrorPriorityComputer(value_representation=val_rep)
    return LossPipeline(modules, priority_computer=priority_computer)


@register_agent("muzero")
def build_muzero(
    config: Any, agent_network: Any, device: torch.device
) -> Dict[str, Any]:
    # 1. Losses
    loss_pipeline = build_muzero_loss_pipeline(config, agent_network, device)

    # 2. Setup Optimizers and Schedulers
    # Factory create_opt logic for non-PPO:
    from torch.optim.adam import Adam
    from torch.optim.sgd import SGD

    optimizers = {}
    lr_schedulers = {}

    def create_opt(params, sub_config_parent):
        opt_cls = sub_config_parent.optimizer
        if opt_cls == Adam:
            return Adam(
                params=params,
                lr=config.learning_rate,
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay,
            )
        elif opt_cls == SGD:
            return SGD(
                params=params,
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        else:
            return opt_cls(params, lr=config.learning_rate)

    opt = create_opt(agent_network.parameters(), config)
    optimizers["default"] = opt
    lr_schedulers["default"] = get_lr_scheduler(opt, config)

    # 3. Callbacks
    callbacks = []
    if getattr(config, "use_noisy_net", False):
        callbacks.append(ResetNoiseCallback())

    # 4. Target Builder
    builders = [
        TrajectoryGradientScaleBuilder(unroll_steps=config.unroll_steps),
        MuZeroTargetBuilder(unroll_steps=config.unroll_steps),
    ]
    if getattr(config, "consistency_loss_factor", 0) > 0:
        builders.append(LatentConsistencyBuilder())

    target_builder = TargetBuilderPipeline(builders)

    return {
        "loss_pipeline": loss_pipeline,
        "optimizers": optimizers,
        "lr_schedulers": lr_schedulers,
        "callbacks": callbacks,
        "target_builder": target_builder,
        "observation_dtype": torch.float32,
    }
