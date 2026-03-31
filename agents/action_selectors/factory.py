from typing import Dict, Any
from agents.action_selectors.selectors import (
    CategoricalSelector,
    EpsilonGreedySelector,
    ArgmaxSelector,
    BaseActionSelector,
)
from agents.action_selectors.decorators import PPODecorator, TemperatureSelector


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
    }

    @classmethod
    def create(cls, config: Any) -> BaseActionSelector:
        """
        Builds the selector chain.
        Expects a dict or SelectorConfig object.
        """
        # 1. Resolve Base Config
        if hasattr(config, "base"):
            base_cfg = config.base
        else:
            base_cfg = config.get("base", {})

        if hasattr(base_cfg, "type"):
            base_type = base_cfg.type
            base_kwargs = getattr(base_cfg, "kwargs", {})
        else:
            base_type = base_cfg.get("type")
            base_kwargs = base_cfg.get("kwargs", {})

        if base_type not in cls.REGISTRY:
            if base_type is None:
                raise ValueError("Selector config must specify a 'base' type.")
            raise ValueError(f"Unknown selector type: {base_type}")

        # Instantiate Base Selector
        if base_type == "categorical":
            selector = CategoricalSelector(**base_kwargs)
        elif base_type == "epsilon_greedy":
            selector = EpsilonGreedySelector(**base_kwargs)
        elif base_type == "argmax":
            selector = ArgmaxSelector()
        else:
            # Fallback for any other registered types
            selector = cls.REGISTRY[base_type](**base_kwargs)

        # 2. Wrap with Decorators
        if hasattr(config, "decorators"):
            decorators_cfg = config.decorators
        else:
            decorators_cfg = config.get("decorators", [])

        for dec_cfg in decorators_cfg:
            if hasattr(dec_cfg, "type"):
                dec_type = dec_cfg.type
                dec_kwargs = getattr(dec_cfg, "kwargs", {})
            else:
                dec_type = dec_cfg.get("type")
                dec_kwargs = dec_cfg.get("kwargs", {})

            if dec_type not in cls.REGISTRY:
                raise ValueError(f"Unknown decorator type: {dec_type}")

            if dec_type == "ppo_injector":
                selector = PPODecorator(inner_selector=selector)
            elif dec_type == "temperature":
                # TemperatureSelector requires schedule_config
                # Try to find it in dec_cfg or dec_kwargs
                schedule_config = None
                if hasattr(dec_cfg, "temperature_schedule"):
                    schedule_config = dec_cfg.temperature_schedule
                elif isinstance(dec_cfg, dict):
                    schedule_config = dec_cfg.get("temperature_schedule")
                
                if schedule_config is None:
                    schedule_config = dec_kwargs.get("temperature_schedule")
                
                if schedule_config is None:
                     raise ValueError("TemperatureSelector requires 'temperature_schedule' in config.")
                
                selector = TemperatureSelector(
                    inner_selector=selector,
                    schedule_config=schedule_config
                )
            else:
                selector = cls.REGISTRY[dec_type](inner_selector=selector, **dec_kwargs)

        return selector
