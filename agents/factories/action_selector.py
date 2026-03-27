from typing import Dict, Any
from agents.action_selectors.selectors import (
    ArgmaxSelector,
    BaseActionSelector,
    LegalMovesMaskDecorator,
)
from agents.action_selectors.decorators import PPODecorator, TemperatureSelector
from agents.action_selectors.selectors import CategoricalSelector, EpsilonGreedySelector


class SelectorFactory:
    """Dynamically builds an Action Selector chain from a config."""

    REGISTRY = {
        # Base Selectors
        "categorical": CategoricalSelector,
        "epsilon_greedy": EpsilonGreedySelector,
        "argmax": ArgmaxSelector,
        # Decorators
        "ppo_injector": PPODecorator,
        "temperature": TemperatureSelector,
        "legal_moves_mask": LegalMovesMaskDecorator,
    }

    @classmethod
    def create(cls, config: Dict[str, Any]) -> BaseActionSelector:
        """
        Builds the selector chain.
        Expects a dict like:
        {
            "base": {"type": "categorical", "kwargs": {"exploration": True}},
            "decorators": [{"type": "temperature", "kwargs": {...}}]
        }
        """
        # 1. Instantiate the Base Selector
        # Handle if config itself is a SelectorConfig object or a plain dict
        if hasattr(config, "base"):
            base_cfg = config.base
        else:
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

        # 1.5 Automatically wrap with LegalMovesMaskDecorator (MANDATORY)
        # Since base selectors (Argmax, Categorical, EpsilonGreedy) no longer handle masks,
        # we must ensure the decorator is present by default. We only skip if explicitly
        # listed in the decorators list to avoid double-masking.
        if hasattr(config, "decorators"):
            decorators_cfg = config.decorators
        else:
            decorators_cfg = config.get("decorators", [])

        has_mask_dec = any(
            (getattr(d, "type", d.get("type")) == "legal_moves_mask")
            for d in decorators_cfg
        )
        if not has_mask_dec:
            selector = LegalMovesMaskDecorator(inner_selector=selector)

        # 2. Recursively wrap it with Decorators (Inside-Out)
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
