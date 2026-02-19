from typing import Dict, Any
from agents.action_selectors.selectors import (
    CategoricalSelector,
    EpsilonGreedySelector,
    ArgmaxSelector,
    BaseActionSelector,
)
from agents.action_selectors.decorators import PPODecorator, MCTSDecorator


class SelectorFactory:
    """Dynamically builds an Action Selector chain from a config."""

    REGISTRY = {
        # Base Selectors
        "categorical": CategoricalSelector,
        "epsilon_greedy": EpsilonGreedySelector,
        "argmax": ArgmaxSelector,
        # Decorators
        "ppo_injector": PPODecorator,
        "mcts": MCTSDecorator,
    }

    @classmethod
    def create(cls, config: Dict[str, Any]) -> BaseActionSelector:
        """
        Builds the selector chain.
        Expects a dict like:
        {
            "base": {"type": "categorical", "kwargs": {"exploration": True}},
            "decorators": [{"type": "ppo_injector", "kwargs": {}}]
        }
        """
        # 1. Instantiate the Base Selector
        base_cfg = config.get("base", {})
        # Handle if base_cfg is a Config object or dict
        if hasattr(base_cfg, "type"):
            base_type = base_cfg.type
            base_kwargs = getattr(base_cfg, "kwargs", {})
        else:
            base_type = base_cfg.get("type")
            base_kwargs = base_cfg.get("kwargs", {})

        if base_type not in cls.REGISTRY:
            # Fallback or error
            if base_type is None:
                raise ValueError("Selector config must specify a 'base' type.")
            raise ValueError(f"Unknown selector type: {base_type}")

        # Create the inner-most selector
        selector = cls.REGISTRY[base_type](**base_kwargs)

        # 2. Recursively wrap it with Decorators (Inside-Out)
        decorators_cfg = config.get("decorators", [])
        for dec_cfg in decorators_cfg:
            # Handle if dec_cfg is a Config object or dict
            if hasattr(dec_cfg, "type"):
                dec_type = dec_cfg.type
                dec_kwargs = getattr(dec_cfg, "kwargs", {})
            else:
                dec_type = dec_cfg.get("type")
                dec_kwargs = dec_cfg.get("kwargs", {})

            if dec_type not in cls.REGISTRY:
                raise ValueError(f"Unknown decorator type: {dec_type}")

            decorator_class = cls.REGISTRY[dec_type]

            # Wrap the current selector!
            selector = decorator_class(inner_selector=selector, **dec_kwargs)

        # Return the fully wrapped, ready-to-use pipeline
        return selector
