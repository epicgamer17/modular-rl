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
from agents.learners.target_builders import DQNTargetBuilder, MuZeroTargetBuilder
from agents.learners.callbacks import (
    PriorityUpdaterCallback,
    WeightBroadcastCallback,
    PPOEarlyStoppingCallback,
    TargetNetworkSyncCallback,
    ResetNoiseCallback,
    LatentMetricsCallback,
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

from agents.registries.base import AGENT_REGISTRY, register_agent


def build_loss_pipeline(
    config: Any, agent_network: ModularAgentNetwork, device: torch.device
) -> LossPipeline:
    """
    DEPRECATED: Use the registry-based build_universal_learner instead.
    Configures the loss pipeline based on the agent configuration.

    Args:
        config: The agent configuration object.
        agent_network: The modular agent network.
        device: The torch device.

    Returns:
        A LossPipeline containing the appropriate loss modules.
    """
    agent_type = config.agent_type

    # This logic is now moved to algorithm-specific builders.
    # We provide a temporary fallback or redirection if needed.
    from agents.registries.ppo import build_ppo_loss_pipeline
    from agents.registries.muzero import build_muzero_loss_pipeline
    from agents.registries.rainbow import build_rainbow_loss_pipeline
    from agents.registries.supervised import build_supervised_loss_pipeline

    if agent_type == "muzero":
        return build_muzero_loss_pipeline(config, agent_network, device)
    elif agent_type == "ppo":
        return build_ppo_loss_pipeline(config, agent_network, device)
    elif agent_type == "rainbow":
        return build_rainbow_loss_pipeline(config, agent_network, device)
    elif agent_type == "supervised":
        return build_supervised_loss_pipeline(config, agent_network, device)

    return LossPipeline([])


def build_universal_learner(
    config: Any,
    agent_network: ModularAgentNetwork,
    device: torch.device,
    priority_update_fn: Optional[Callable] = None,
    weight_broadcast_fn: Optional[Callable] = None,
) -> UniversalLearner:
    """
    Establishes a clean factory interface for assembling a UniversalLearner.
    Wires together the network, device, loss pipeline, and all necessary callbacks.
    Uses the AGENT_REGISTRY to find algorithm-specific builders.

    Args:
        config: The agent configuration object.
        agent_network: The modular agent network instance.
        device: Torch device (cpu/cuda/mps).
        priority_update_fn: Optional callable to update PER priorities.
        weight_broadcast_fn: Optional callable to broadcast weights to workers.

    Returns:
        Constructed UniversalLearner instance.
    """
    # 1. Deduce agent type
    agent_type = config.agent_type

    # 2. Lazy imports to avoid circular dependencies and ensure registration
    if agent_type == "ppo":
        import agents.registries.ppo  # noqa: F401
    elif agent_type == "muzero":
        import agents.registries.muzero  # noqa: F401
    elif agent_type == "rainbow":
        import agents.registries.rainbow  # noqa: F401
    elif agent_type == "supervised":
        import agents.registries.supervised  # noqa: F401

    # 3. Get builder from registry
    assert (
        agent_type in AGENT_REGISTRY
    ), f"Agent type '{agent_type}' not found in AGENT_REGISTRY. Registered: {list(AGENT_REGISTRY.keys())}"
    builder_fn = AGENT_REGISTRY[agent_type]

    # 4. Call algorithm-specific builder
    # The builder is responsible for setup of losses, optimizers, and algo-specific callbacks.
    builder_output = builder_fn(config, agent_network, device)

    # Unpack builder output (supporting flexibility in what builders return)
    # Recommended return: (loss_pipeline, optimizers, lr_schedulers, callbacks, target_builder, observation_dtype)
    loss_pipeline = builder_output["loss_pipeline"]
    optimizers = builder_output["optimizers"]
    lr_schedulers = builder_output["lr_schedulers"]
    callbacks = builder_output.get("callbacks", [])
    target_builder = builder_output.get("target_builder", None)
    observation_dtype = builder_output.get("observation_dtype", torch.float32)

    # 5. Add standard infrastructure callbacks
    if priority_update_fn:
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

    # 6. Construct and return UniversalLearner
    return UniversalLearner(
        config=config,
        agent_network=agent_network,
        device=device,
        num_actions=agent_network.num_actions,
        observation_dimensions=agent_network.input_shape,
        observation_dtype=observation_dtype,
        loss_pipeline=loss_pipeline,
        optimizer=optimizers,
        lr_scheduler=lr_schedulers,
        callbacks=callbacks,
        clipnorm=config.clipnorm,
        target_builder=target_builder,
    )
