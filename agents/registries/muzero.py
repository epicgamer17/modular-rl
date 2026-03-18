import torch
from typing import Any, Dict, List, Tuple
from agents.registries.base import register_agent
from losses.losses import (
    LossPipeline,
    ValueLoss,
    PolicyLoss,
    RewardLoss,
    ToPlayLoss,
    ConsistencyLoss,
    ChanceQLoss,
    SigmaLoss,
    VQVAECommitmentLoss,
    SpecificLossPriority,
)
from agents.learner.callbacks import (
    ResetNoiseCallback,
)
from modules.utils import get_lr_scheduler
from agents.learner.target_builders import (
    TargetBuilderPipeline,
    TrajectoryGradientScaleBuilder,
    LatentConsistencyBuilder,
)

from losses.representations import ClassificationRepresentation


def build_muzero_loss_pipeline(config, agent_network, device):
    modules = [
        ValueLoss(config, device),
        PolicyLoss(
            config,
            device,
            ClassificationRepresentation(num_classes=config.game.num_actions),
        ),
        RewardLoss(config, device),
    ]
    if getattr(config.game, "num_players", 1) > 1:
        modules.append(
            ToPlayLoss(
                config,
                device,
                ClassificationRepresentation(num_classes=config.game.num_players),
            )
        )
    if getattr(config, "consistency_loss_factor", 0) > 0:
        modules.append(ConsistencyLoss(config, device, agent_network))
    if getattr(config, "stochastic", False):
        modules.extend(
            [
                ChanceQLoss(config, device),
                SigmaLoss(config, device),
                VQVAECommitmentLoss(config, device),
            ]
        )
    priority_computer = SpecificLossPriority(loss_name="ValueLoss")
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
    callbacks = []
    if getattr(config, "use_noisy_net", False):
        callbacks.append(ResetNoiseCallback())

    # 4. Target Builder
    builders = [TrajectoryGradientScaleBuilder(unroll_steps=config.unroll_steps)]
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
