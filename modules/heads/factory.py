from typing import Tuple, Dict, Type, Any
from torch import nn
from .to_play import ToPlayHead
from configs.modules.heads.to_play import ToPlayHeadConfig
from configs.modules.heads.base import HeadConfig
from configs.modules.architecture_config import ArchitectureConfig


class HeadFactory:
    """Factory to create Head modules based on their configuration."""

    _heads: Dict[Type[HeadConfig], Type[nn.Module]] = {
        ToPlayHeadConfig: ToPlayHead,
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
            )

        return head_cls(arch_config, input_shape, config, **kwargs)
