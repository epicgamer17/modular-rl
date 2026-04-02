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

from agents.learner.base import UniversalLearner
from agents.learner.target_builders import TemporalDifferenceBuilder
from agents.learner.callbacks import (
    PriorityUpdaterCallback,
    WeightBroadcastCallback,
    MetricEarlyStopCallback,
    TargetNetworkSyncCallback,
    ResetNoiseCallback,
    EpsilonGreedySchedulerCallback,
)
from agents.learner.losses import LossPipeline
from modules.utils import get_lr_scheduler

if TYPE_CHECKING:
    from modules.agent_nets.modular import ModularAgentNetwork


def build_loss_pipeline(
    config: Any,
    agent_network: ModularAgentNetwork,
    device: torch.device,
    shape_validator: Optional[ShapeValidator] = None,
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
    agent_type = getattr(config, "agent_type", None)
    if agent_type == "supervised":
        from agents.registries.supervised import build_supervised_loss_pipeline

        return build_supervised_loss_pipeline(
            agent_network=agent_network,
            device=device,
            minibatch_size=config.minibatch_size,
            num_actions=agent_network.num_actions,
            shape_validator=shape_validator,
        )

    raise ValueError(f"Unknown agent type: {agent_type}")


def build_universal_learner(
    config: Any,
    agent_network: ModularAgentNetwork,
    device: torch.device,
    target_agent_network: Optional[nn.Module] = None,
    priority_update_fn: Optional[Callable] = None,
    set_beta_fn: Optional[Callable] = None,
    per_beta_schedule: Optional[Any] = None,
    epsilon_schedule: Optional[Any] = None,
    weight_broadcast_fn: Optional[Callable] = None,
    extra_callbacks: Optional[List[Callback]] = None,
) -> UniversalLearner:
    """
    Establishes a clean factory interface for assembling a UniversalLearner.
    Wires together the network, device, loss pipeline, and all necessary callbacks.

    Args:
        config: The agent configuration object.
        agent_network: The modular agent network instance.
        device: Torch device (cpu/cuda/mps).
        target_agent_network: Optional target network for TD-based algorithms.
        priority_update_fn: Optional callable to update PER priorities.
        weight_broadcast_fn: Optional callable to broadcast weights to workers.

    Returns:
        Constructed UniversalLearner instance.
    """
    from agents.learner.losses.shape_validator import ShapeValidator

    # 0. Instantiate ShapeValidator externally
    shape_validator = ShapeValidator(
        minibatch_size=config.minibatch_size,
        num_actions=agent_network.num_actions,
        unroll_steps=getattr(config, "unroll_steps", 0),
        atom_size=getattr(config, "atom_size", 1),
        support_range=getattr(config, "support_range", None),
    )

    callbacks = []
    optimizers = {}
    lr_schedulers = {}
    observation_dtype = torch.float32

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

    # Ensure agent type is explicitly defined
    agent_type = getattr(config, "agent_type", None)
    if agent_type is None:
        raise ValueError("config.agent_type must be explicitly defined.")
    target_builder = None
    observation_dtype = torch.float32

    if agent_type == "supervised":
        from agents.registries.supervised import build_supervised

        supervised_components = build_supervised(config, agent_network, device)
        optimizers = supervised_components["optimizers"]
        lr_schedulers = supervised_components["lr_schedulers"]
        callbacks.extend(supervised_components.get("callbacks", []))
        observation_dtype = supervised_components.get(
            "observation_dtype", torch.float32
        )

    # 2. Setup generic algorithmic callbacks
    # Priority Updates (PER) are already handled above if priority_update_fn was provided.
    # Target Network Sync is handled by specific registries (like Rainbow) that require it.

    # Epsilon Scheduling
    if epsilon_schedule is not None:
        callbacks.append(EpsilonGreedySchedulerCallback(epsilon_schedule))

    # Weight Broadcasting
    if weight_broadcast_fn is not None:
        callbacks.append(WeightBroadcastCallback(weight_broadcast_fn))

    return UniversalLearner(
        agent_network=agent_network,
        device=device,
        num_actions=agent_network.num_actions,
        observation_dimensions=agent_network.input_shape,
        observation_dtype=observation_dtype,
        loss_pipeline=build_loss_pipeline(
            config, agent_network, device, shape_validator=shape_validator
        ),
        optimizer=optimizers,
        lr_scheduler=lr_schedulers,
        callbacks=callbacks,
        shape_validator=shape_validator,
        clipnorm=getattr(config, "clipnorm", None),
        gradient_accumulation_steps=getattr(config, "gradient_accumulation_steps", 1),
        max_grad_norm=getattr(config, "max_grad_norm", None),
        target_builder=target_builder,
    )
