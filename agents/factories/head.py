from typing import Tuple, Dict, Type, Any, Optional
from torch import nn
from modules.heads.to_play import ToPlayHead
from modules.heads.base import BaseHead
from modules.heads.reward import RewardHead, ValuePrefixRewardHead
from modules.heads.value import ValueHead
from modules.heads.policy import PolicyHead
from modules.heads.chance_probability import ChanceProbabilityHead
from modules.heads.continuation import ContinuationHead
from modules.heads.observation import ObservationHead
from modules.heads.latent_consistency import SimSiamProjectorHead
from modules.heads.q import QHead, DuelingQHead
from configs.modules.heads.to_play import ToPlayHeadConfig
from configs.modules.heads.value import ValueHeadConfig
from configs.modules.heads.reward import RewardHeadConfig, ValuePrefixRewardHeadConfig
from configs.modules.heads.policy import PolicyHeadConfig
from configs.modules.heads.chance_probability import ChanceProbabilityHeadConfig
from configs.modules.heads.continuation import ContinuationHeadConfig
from configs.modules.heads.observation import ObservationHeadConfig
from configs.modules.heads.latent_consistency import SimSiamProjectorConfig
from configs.modules.heads.q import QHeadConfig, DuelingQHeadConfig
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
        SimSiamProjectorConfig: SimSiamProjectorHead,
        QHeadConfig: QHead,
        DuelingQHeadConfig: DuelingQHead,
    }

    @classmethod
    def create(
        cls,
        config: HeadConfig,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        name: Optional[str] = None,
        **kwargs,
    ) -> nn.Module:
        config_type = type(config)
        if config_type not in cls._heads:
            raise ValueError(
                f"No head module registered for config type: {config_type}"
            )

        head_cls = cls._heads[config_type]

        # 1. Automatic Representation Resolution
        # The factory handles building the representation if the config specifies it.
        # This prevents the router (AgentNetwork/WorldModel) from having to know
        # about the heads' mathematical format.
        representation = kwargs.get("representation")
        if (
            representation is None
            and hasattr(config, "output_strategy")
            and config.output_strategy is not None
        ):
            from agents.learner.losses.representations import get_representation

            representation = get_representation(config.output_strategy)

        # 2. ToPlayHead Specialization
        noisy_sigma = arch_config.noisy_sigma

        if isinstance(config, ToPlayHeadConfig):
            num_players = kwargs.get("num_players", getattr(config, "num_players", None))
            if num_players is None:
                raise ValueError(
                    "ToPlayHead requires num_players (either in config or passed to factory)"
                )
            return ToPlayHead(
                input_shape=input_shape,
                num_players=num_players,
                neck_config=config.neck,
                representation=representation,
                noisy_sigma=noisy_sigma,
                name=name,
                input_source=config.input_source,
            )

        # PolicyHead
        if isinstance(config, PolicyHeadConfig):
            return PolicyHead(
                input_shape=input_shape,
                neck_config=config.neck,
                representation=representation,
                noisy_sigma=noisy_sigma,
                name=name,
                input_source=config.input_source,
            )

        # ChanceProbabilityHead needs num_chance_codes
        if isinstance(config, ChanceProbabilityHeadConfig):
            num_chance_codes = kwargs.get("num_chance_codes")
            if num_chance_codes is None:
                raise ValueError(
                    "ChanceProbabilityHead requires num_chance_codes to be passed."
                )
            return ChanceProbabilityHead(
                input_shape=input_shape,
                num_chance_codes=num_chance_codes,
                neck_config=config.neck,
                noisy_sigma=noisy_sigma,
                name=name,
                input_source=config.input_source,
            )

        # ValuePrefixRewardHead takes specific config
        if isinstance(config, ValuePrefixRewardHeadConfig):
            return ValuePrefixRewardHead(
                input_shape=input_shape,
                representation=representation,  # Representation is resolved by factory
                lstm_hidden_size=config.lstm_hidden_size,
                lstm_horizon_len=config.lstm_horizon_len,
                neck_config=config.neck,
                noisy_sigma=noisy_sigma,
                name=name,
                input_source=config.input_source,
            )

        # SimSiamProjectorHead takes specific config
        if isinstance(config, SimSiamProjectorConfig):
            return SimSiamProjectorHead(
                input_shape=input_shape,
                proj_hidden_dim=config.proj_hidden_dim,
                proj_output_dim=config.proj_output_dim,
                pred_hidden_dim=config.pred_hidden_dim,
                pred_output_dim=config.pred_output_dim,
                representation=representation,
                neck_config=config.neck,
                noisy_sigma=noisy_sigma,
                name=name,
                input_source=config.input_source,
            )

        # QHead
        if isinstance(config, QHeadConfig):
            num_actions = kwargs.get("num_actions")
            if num_actions is None:
                raise ValueError("QHead requires num_actions to be passed.")
            return QHead(
                input_shape=input_shape,
                representation=representation,
                hidden_backbone_config=config.hidden_backbone,
                num_actions=num_actions,
                neck_config=config.neck,
                noisy_sigma=noisy_sigma,
                name=name,
                input_source=config.input_source,
            )

        # DuelingQHead
        if isinstance(config, DuelingQHeadConfig):
            num_actions = kwargs.get("num_actions")
            if num_actions is None:
                raise ValueError("DuelingQHead requires num_actions to be passed.")
            return DuelingQHead(
                input_shape=input_shape,
                representation=representation,
                value_hidden_backbone_config=config.value_hidden_backbone,
                advantage_hidden_backbone_config=config.advantage_hidden_backbone,
                num_actions=num_actions,
                neck_config=config.neck,
                noisy_sigma=noisy_sigma,
                name=name,
                input_source=config.input_source,
            )

        return head_cls(
            input_shape=input_shape,
            representation=representation,
            neck_config=config.neck,
            noisy_sigma=noisy_sigma,
            name=name,
            input_source=config.input_source,
        )
