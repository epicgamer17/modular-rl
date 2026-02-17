from typing import Tuple, Dict, Type, Any, Optional
from torch import nn
from .to_play import ToPlayHead
from .base import BaseHead
from .reward import RewardHead, ValuePrefixRewardHead
from .value import ValueHead
from .policy import PolicyHead
from .chance_probability import ChanceProbabilityHead
from configs.modules.heads.to_play import ToPlayHeadConfig
from configs.modules.heads.value import ValueHeadConfig
from configs.modules.heads.reward import RewardHeadConfig, ValuePrefixRewardHeadConfig
from configs.modules.heads.policy import PolicyHeadConfig
from configs.modules.heads.chance_probability import ChanceProbabilityHeadConfig
from configs.modules.heads.base import HeadConfig
from configs.modules.architecture_config import ArchitectureConfig


class HeadFactory:
    """Factory to create Head modules based on their configuration."""

    _heads: Dict[Type[HeadConfig], Type[nn.Module]] = {
        ToPlayHeadConfig: ToPlayHead,
        ValueHeadConfig: ValueHead,
        RewardHeadConfig: RewardHead,
        ValuePrefixRewardHeadConfig: ValuePrefixRewardHead,
        PolicyHeadConfig: PolicyHead,
        ChanceProbabilityHeadConfig: ChanceProbabilityHead,
    }

    @classmethod
    def create(
        cls,
        config: HeadConfig,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        **kwargs,
    ) -> nn.Module:
        config_type = type(config)
        if config_type not in cls._heads:
            raise ValueError(
                f"No head module registered for config type: {config_type}"
            )

        head_cls = cls._heads[config_type]

        # Merge kwargs with config values if needed, or pass them explicitly
        # For ToPlayHead, we might need num_players
        if isinstance(config, ToPlayHeadConfig):
            num_players = kwargs.get("num_players", config.num_players)
            if num_players is None:
                raise ValueError(
                    "ToPlayHead requires num_players (either in config or passed to factory)"
                )
            return ToPlayHead(
                arch_config=arch_config,
                input_shape=input_shape,
                num_players=num_players,
                neck_config=config.neck,
                strategy=kwargs.get("strategy"),
            )

        # PolicyHead needs num_actions OR output_size
        if isinstance(config, PolicyHeadConfig):
            # Try to get either num_actions (discrete) or output_size (continuous)
            num_actions = kwargs.get("num_actions")
            output_size = kwargs.get("output_size")

            return PolicyHead(
                arch_config=arch_config,
                input_shape=input_shape,
                num_actions=num_actions,
                output_size=output_size,
                neck_config=config.neck,
                strategy=kwargs.get("strategy"),
            )

        # ChanceProbabilityHead needs num_chance_codes
        if isinstance(config, ChanceProbabilityHeadConfig):
            num_chance_codes = kwargs.get("num_chance_codes")
            if num_chance_codes is None:
                raise ValueError(
                    "ChanceProbabilityHead requires num_chance_codes to be passed."
                )
            return ChanceProbabilityHead(
                arch_config=arch_config,
                input_shape=input_shape,
                num_chance_codes=num_chance_codes,
                neck_config=config.neck,
            )

        # ValuePrefixRewardHead takes specific config
        if isinstance(config, ValuePrefixRewardHeadConfig):
            return ValuePrefixRewardHead(
                arch_config=arch_config,
                input_shape=input_shape,
                strategy=kwargs.get(
                    "strategy"
                ),  # Strategy is usually created outside and passed in
                config=config,
                neck_config=config.neck,
            )

        return head_cls(
            arch_config=arch_config,
            input_shape=input_shape,
            strategy=kwargs.get("strategy"),
            neck_config=config.neck,
        )
