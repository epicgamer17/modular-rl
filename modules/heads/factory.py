from typing import Tuple, Dict, Type, Any, Optional
from torch import nn
from .to_play import ToPlayHead
from .base import BaseHead
from .reward import RewardHead, ValuePrefixRewardHead
from .value import ValueHead
from .policy import PolicyHead
from .chance_probability import ChanceProbabilityHead
from .continuation import ContinuationHead
from .observation import ObservationHead
from .latent_consistency import LatentConsistencyHead
from configs.modules.heads.to_play import ToPlayHeadConfig
from configs.modules.heads.value import ValueHeadConfig
from configs.modules.heads.reward import RewardHeadConfig, ValuePrefixRewardHeadConfig
from configs.modules.heads.policy import PolicyHeadConfig
from configs.modules.heads.chance_probability import ChanceProbabilityHeadConfig
from configs.modules.heads.continuation import ContinuationHeadConfig
from configs.modules.heads.observation import ObservationHeadConfig
from configs.modules.heads.latent_consistency import LatentConsistencyHeadConfig
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
        ContinuationHeadConfig: ContinuationHead,
        ObservationHeadConfig: ObservationHead,
        LatentConsistencyHeadConfig: LatentConsistencyHead,
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

        # 1. Build Neck (Pre-instantiated)
        neck = None
        if hasattr(config, "neck") and config.neck is not None:
            from modules.backbones.factory import BackboneFactory

            neck = BackboneFactory.create(config.neck, input_shape)

        # 2. Extract shared parameters from arch_config
        noisy_sigma = getattr(arch_config, "noisy_sigma", 0.0)

        # 3. Instantiate Head with explicit parameters
        if isinstance(config, ToPlayHeadConfig):
            num_players = kwargs.get("num_players", config.num_players)
            if num_players is None:
                raise ValueError(
                    "ToPlayHead requires num_players (either in config or passed to factory)"
                )
            return ToPlayHead(
                input_shape=input_shape,
                num_players=num_players,
                neck=neck,
                noisy_sigma=noisy_sigma,
                representation=kwargs.get("representation"),
            )

        if isinstance(config, PolicyHeadConfig):
            return PolicyHead(
                input_shape=input_shape,
                representation=kwargs.get("representation"),
                neck=neck,
                noisy_sigma=noisy_sigma,
            )

        if isinstance(config, ChanceProbabilityHeadConfig):
            num_chance_codes = kwargs.get("num_chance_codes")
            if num_chance_codes is None:
                raise ValueError(
                    "ChanceProbabilityHead requires num_chance_codes to be passed."
                )
            return ChanceProbabilityHead(
                input_shape=input_shape,
                num_chance_codes=num_chance_codes,
                neck=neck,
                noisy_sigma=noisy_sigma,
            )

        if isinstance(config, ValuePrefixRewardHeadConfig):
            return ValuePrefixRewardHead(
                input_shape=input_shape,
                representation=kwargs.get("representation"),
                lstm_hidden_size=config.lstm_hidden_size,
                lstm_horizon_len=config.lstm_horizon_len,
                neck=neck,
                noisy_sigma=noisy_sigma,
            )

        if isinstance(config, LatentConsistencyHeadConfig):
            return LatentConsistencyHead(
                input_shape=input_shape,
                representation=kwargs.get("representation"),
                neck=neck,
                noisy_sigma=noisy_sigma,
                projection_dim=config.projection_dim,
            )

        # Fallback for standard heads (ValueHead, RewardHead, etc.)
        return head_cls(
            input_shape=input_shape,
            representation=kwargs.get("representation"),
            neck=neck,
            noisy_sigma=noisy_sigma,
        )
