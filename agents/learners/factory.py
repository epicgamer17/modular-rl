from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)
import torch
from torch import nn

from agents.learners.base import UniversalLearner
from agents.learners.callbacks import (
    PriorityUpdaterCallback,
    WeightBroadcastCallback,
    PPOEarlyStoppingCallback,
    TargetNetworkSyncCallback,
    ResetNoiseCallback,
    LatentMetricsCallback,
    StochasticMetricsCallback,
)
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
    PPOPolicyLoss,
    PPOValueLoss,
    StandardDQNLoss,
    C51Loss,
    ImitationLoss,
)
from modules.utils import get_lr_scheduler

if TYPE_CHECKING:
    from modules.agent_nets.modular import ModularAgentNetwork


def build_loss_pipeline(
    config: Any, agent_network: ModularAgentNetwork, device: torch.device
) -> LossPipeline:
    """
    Configures the loss pipeline based on the agent configuration.

    Args:
        config: The agent configuration object.
        agent_network: The modular agent network.
        device: The torch device.

    Returns:
        A LossPipeline containing the appropriate loss modules.
    """
    # Use agent_type if available, otherwise deduce from config fields
    agent_type = getattr(config, "agent_type", None)
    if agent_type is None:
        if hasattr(config, "clip_param"):
            agent_type = "ppo"
        elif hasattr(config, "unroll_steps"):
            agent_type = "muzero"
        elif hasattr(config, "dueling") or hasattr(config, "atom_size"):
            agent_type = "rainbow"
        else:
            agent_type = "supervised"

    modules = []

    if agent_type == "muzero":
        modules = [
            ValueLoss(config, device),
            PolicyLoss(config, device),
            RewardLoss(config, device),
        ]
        if getattr(config.game, "num_players", 1) > 1:
            modules.append(ToPlayLoss(config, device))
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

    elif agent_type == "ppo":
        modules = [
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

    elif agent_type == "rainbow":
        from agents.action_selectors.selectors import ArgmaxSelector

        selector = ArgmaxSelector()
        td_loss_module = (
            C51Loss(config=config, device=device, action_selector=selector)
            if getattr(config, "atom_size", 1) > 1
            else StandardDQNLoss(config=config, device=device, action_selector=selector)
        )
        modules = [td_loss_module]

    elif agent_type == "supervised":
        modules = [ImitationLoss(config, device, agent_network.num_actions)]

    return LossPipeline(modules)


def build_universal_learner(
    config: Any,
    agent_network: ModularAgentNetwork,
    device: torch.device,
    batch_iterator: Iterable,  # Notice: No Replay Buffer here!
    priority_update_fn: Optional[Callable] = None,
    weight_broadcast_fn: Optional[Callable] = None,
) -> UniversalLearner:
    """
    Establishes a clean factory interface for assembling a UniversalLearner.
    Wires together the network, device, loss pipeline, and all necessary callbacks.

    Args:
        config: The agent configuration object.
        agent_network: The modular agent network instance.
        device: Torch device (cpu/cuda/mps).
        batch_iterator: An iterable yielding transition batches.
        priority_update_fn: Optional callable to update PER priorities.
        weight_broadcast_fn: Optional callable to broadcast weights to workers.

    Returns:
        Constructed UniversalLearner instance.
    """

    callbacks = []

    if priority_update_fn:
        # We assume if a priority_update_fn is provided, it's a method of a buffer that also has set_beta
        # This is a bit of a leap, but in this framework buffers that support PER follow this pattern.
        # If not, it will fail-fast as requested.
        set_beta_fn = getattr(priority_update_fn.__self__, "set_beta", None)
        from utils.schedule import create_schedule

        per_beta_schedule_config = getattr(config, "per_beta_schedule", None)
        per_beta_schedule = create_schedule(per_beta_schedule_config)

        callbacks.append(
            PriorityUpdaterCallback(
                priority_update_fn=priority_update_fn,
                set_beta_fn=set_beta_fn,
                per_beta_schedule=per_beta_schedule,
            )
        )

    if weight_broadcast_fn:
        callbacks.append(WeightBroadcastCallback(weight_broadcast_fn))

    # Deduce agent type
    agent_type = getattr(config, "agent_type", None)
    if agent_type is None:
        if hasattr(config, "clip_param"):
            agent_type = "ppo"
        elif hasattr(config, "unroll_steps"):
            agent_type = "muzero"
        elif hasattr(config, "dueling") or hasattr(config, "atom_size"):
            agent_type = "rainbow"
        else:
            agent_type = "supervised"

    # Append algorithmic callbacks based on config
    if agent_type == "ppo":
        callbacks.append(PPOEarlyStoppingCallback(config.target_kl))

    if agent_type == "muzero":
        callbacks.append(ResetNoiseCallback())
        if getattr(config, "stochastic", False):
            callbacks.append(StochasticMetricsCallback())
        if getattr(config, "latent_viz_interval", 0) > 0:
            callbacks.append(
                LatentMetricsCallback(
                    viz_interval=config.latent_viz_interval,
                    viz_method=config.latent_viz_method,
                )
            )

    if agent_type == "rainbow":
        callbacks.append(ResetNoiseCallback())
        # Target network sync is usually added if a target network exists.
        # This factory assumes the target network sync logic is either internal
        # or will be added manually if complex.

    # Setup Optimizers and Schedulers
    from torch.optim.adam import Adam
    from torch.optim.sgd import SGD

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

    if agent_type == "ppo":
        # PPO requires separate optimizers for policy and value heads
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
    else:
        # Default single optimizer
        opt = create_opt(agent_network.parameters(), config)
        optimizers["default"] = opt
        lr_schedulers["default"] = get_lr_scheduler(opt, config)

    return UniversalLearner(
        config=config,
        agent_network=agent_network,
        device=device,
        num_actions=agent_network.num_actions,
        observation_dimensions=agent_network.input_shape,
        observation_dtype=torch.float32,
        loss_pipeline=build_loss_pipeline(config, agent_network, device),
        optimizer=optimizers,
        lr_scheduler=lr_schedulers,
        callbacks=callbacks,
        clip_norm=getattr(config, "clip_norm", getattr(config, "clipnorm", None)),
    )
