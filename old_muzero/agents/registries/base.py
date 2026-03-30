from typing import Dict, Callable, Any

AGENT_REGISTRY: Dict[str, Callable] = {}


def register_agent(name: str):
    """
    Decorator to register an agent builder function.

    Args:
        name: The name of the agent type (e.g., 'ppo', 'muzero').

    Returns:
        The decorator function.
    """

    def decorator(builder_fn):
        AGENT_REGISTRY[name] = builder_fn
        return builder_fn

    return decorator
