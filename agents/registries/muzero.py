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
)
from modules.utils import create_optimizer, get_lr_scheduler
from agents.learner.callbacks import ResetNoiseCallback
from agents.learner.target_builders import (
    TargetBuilderPipeline,
    TrajectoryGradientScaleBuilder,
    LatentConsistencyBuilder,
)


def build_muzero_loss_pipeline(config, agent_network, device):
    world_model = agent_network.components["world_model"]
    modules = [
        ValueLoss(
            config,
            device,
            representation=agent_network.components["value_head"].strategy.representation,
        ),
        PolicyLoss(
            config,
            device,
            representation=agent_network.components["policy_head"].strategy.representation,
        ),
        RewardLoss(
            config,
            device,
            representation=world_model.reward_head.strategy.representation,
        ),
    ]
    if getattr(config.game, "num_players", 1) > 1:
        modules.append(
            ToPlayLoss(
                config,
                device,
                representation=world_model.to_play_head.strategy.representation,
            )
        )
    if getattr(config, "consistency_loss_factor", 0) > 0:
        modules.append(ConsistencyLoss(config, device, agent_network))
    if getattr(config, "stochastic", False):
        modules.extend(
            [
                ChanceQLoss(
                    config,
                    device,
                    representation=agent_network.components[
                        "afterstate_value_head"
                    ].strategy.representation,
                ),
                SigmaLoss(
                    config,
                    device,
                    representation=world_model.sigma_head.strategy.representation,
                ),
                VQVAECommitmentLoss(config, device),
            ]
        )
    return LossPipeline(modules)


@register_agent("muzero")
def build_muzero(
    config: Any, agent_network: Any, device: torch.device
) -> Dict[str, Any]:
    # 1. Losses
    loss_pipeline = build_muzero_loss_pipeline(config, agent_network, device)

    # 2. Optimizers
    opt = create_optimizer(agent_network.parameters(), config)
    optimizers = {"default": opt}
    lr_schedulers = {"default": get_lr_scheduler(opt, config)}

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
