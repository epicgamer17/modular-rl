import torch
from typing import Any, Dict, List, Tuple, Optional, Callable
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
    MetricEarlyStopCallback,
)
from modules.utils import get_lr_scheduler
from agents.learner.target_builders import (
    TargetBuilderPipeline,
    LatentConsistencyBuilder,
    MCTSExtractor,
    SequencePadder,
    SequenceMaskBuilder,
    SequenceInfrastructureBuilder,
    ChanceTargetBuilder,
)


from agents.learner.losses.representations import IdentityRepresentation
from agents.learner.losses.priorities import ExpectedValueErrorPriorityComputer


def build_muzero_loss_pipeline(config, agent_network, device):
    # Extract representations from heads
    val_rep = agent_network.components["behavior_heads"]["state_value"].representation
    pol_rep = agent_network.components["behavior_heads"]["policy_logits"].representation
    rew_rep = (
        agent_network.components["world_model"].heads["reward_logits"].representation
    )
    tp_rep = (
        agent_network.components["world_model"].heads["to_play_logits"].representation
    )

    modules = [
        ValueLoss(
            device=device,
            representation=val_rep,
            loss_fn=config.value_loss_function,
            loss_factor=config.value_loss_factor,
        ),
        PolicyLoss(
            device=device,
            representation=pol_rep,
            loss_fn=config.policy_loss_function,
            loss_factor=config.policy_loss_factor,
        ),
        RewardLoss(
            device=device,
            representation=rew_rep,
            loss_fn=config.reward_loss_function,
            loss_factor=config.reward_loss_factor,
        ),
    ]

    if config.game.num_players > 1:
        modules.append(
            ToPlayLoss(
                device=device,
                representation=tp_rep,
                loss_factor=config.to_play_loss_factor,
            )
        )

    if config.consistency_loss_factor > 0:
        modules.append(
            ConsistencyLoss(
                device=device,
                representation=IdentityRepresentation(),
                agent_network=agent_network,
                loss_factor=config.consistency_loss_factor,
            )
        )

    if config.stochastic:
        as_val_rep = agent_network.components["behavior_heads"][
            "afterstate_value"
        ].representation
        sigma_rep = agent_network.components[
            "world_model"
        ].dynamics_pipeline.sigma_head.representation

        modules.extend(
            [
                ChanceQLoss(
                    device=device,
                    representation=as_val_rep,
                    loss_factor=config.chance_q_loss_factor,
                ),
                SigmaLoss(
                    device=device,
                    representation=sigma_rep,
                    loss_factor=config.sigma_loss_factor,
                ),
                CommitmentLoss(
                    device=device,
                    representation=IdentityRepresentation(),
                ),
            ]
        )

    priority_computer = ExpectedValueErrorPriorityComputer(value_representation=val_rep)
    return LossPipeline(config, modules, priority_computer=priority_computer)


@register_agent("muzero")
def build_muzero(
    config: Any,
    agent_network: Any,
    device: torch.device,
    priority_update_fn: Optional[Callable] = None,
    set_beta_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    # 1. Losses
    loss_pipeline = build_muzero_loss_pipeline(config, agent_network, device)

    # 2. Setup Optimizers and Schedulers
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
    from agents.learner.callbacks import (
        ResetNoiseCallback,
        MetricEarlyStopCallback,
        PriorityUpdaterCallback,
    )
    from utils.schedule import create_schedule

    callbacks = []
    if getattr(config, "use_noisy_net", False):
        callbacks.append(ResetNoiseCallback())
    if getattr(config, "use_early_stopping", False):
        callbacks.append(MetricEarlyStopCallback(threshold=config.early_stopping_kl))

    # Priority Updates (PER)
    if priority_update_fn is not None:
        per_beta_schedule_config = getattr(config, "per_beta_schedule", None)
        per_beta_schedule = create_schedule(per_beta_schedule_config)
        callbacks.append(
            PriorityUpdaterCallback(
                priority_update_fn=priority_update_fn,
                set_beta_fn=set_beta_fn,
                per_beta_schedule=per_beta_schedule,
            )
        )

    # 4. Target Builder
    builders = [
        MCTSExtractor(),
        SequencePadder(unroll_steps=config.unroll_steps),
        SequenceMaskBuilder(),
        SequenceInfrastructureBuilder(unroll_steps=config.unroll_steps),
    ]
    if getattr(config, "consistency_loss_factor", 0) > 0:
        builders.append(LatentConsistencyBuilder())

    if config.stochastic:
        builders.append(ChanceTargetBuilder())

    target_builder = TargetBuilderPipeline(builders)

    return {
        "loss_pipeline": loss_pipeline,
        "optimizers": optimizers,
        "lr_schedulers": lr_schedulers,
        "callbacks": callbacks,
        "target_builder": target_builder,
        "observation_dtype": torch.float32,
    }
