import torch
from typing import Any, Dict, Tuple, Optional
from modules.agent_nets.modular import ModularAgentNetwork
from agents.registries.muzero import build_muzero_network_components
from agents.registries.rainbow import build_rainbow_network_components
from agents.registries.supervised import build_supervised_network_components


def build_modular_agent_network(
    config: Any,
    input_shape: Tuple[int, ...],
    num_actions: int,
    **kwargs,
) -> ModularAgentNetwork:
    """
    Factory function to build a ModularAgentNetwork by delegating component
    instantiation to agent-specific registry functions.
    """
    agent_type = getattr(config, "agent_type", None)

    if agent_type == "muzero":
        build_fn = build_muzero_network_components
    elif agent_type == "rainbow":
        build_fn = build_rainbow_network_components
    elif agent_type == "supervised":
        build_fn = build_supervised_network_components
    else:
        # Fallback to checking config class type if agent_type is missing
        from configs.agents.muzero import MuZeroConfig
        from configs.agents.rainbow_dqn import RainbowConfig
        from configs.agents.supervised import SupervisedConfig

        if isinstance(config, MuZeroConfig):
            build_fn = build_muzero_network_components
        elif isinstance(config, RainbowConfig):
            build_fn = build_rainbow_network_components
        elif isinstance(config, SupervisedConfig):
            build_fn = build_supervised_network_components
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")

    # 1. Delegate component assembly to the registry
    result = build_fn(config, input_shape, num_actions, **kwargs)
    components = result["components"]
    metadata = result["metadata"]

    # 2. Instantiate the lean ModularAgentNetwork container
    return ModularAgentNetwork(
        input_shape=input_shape,
        num_actions=num_actions,
        components=components,
        **metadata,
        **kwargs,
    )
