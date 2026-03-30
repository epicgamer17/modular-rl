from typing import Type, Dict, Tuple, Optional
import torch
from torch import nn
from old_muzero.modules.backbones.resnet import ResNetBackbone
from old_muzero.modules.backbones.denseresnet import DenseResNetBackbone
from old_muzero.modules.backbones.dense import DenseBackbone
from old_muzero.modules.backbones.conv import ConvBackbone
from old_muzero.modules.backbones.recurrent import RecurrentBackbone
from old_muzero.modules.backbones.transformer import TransformerBackbone
from old_muzero.modules.backbones.identity import IdentityBackbone
from old_muzero.modules.backbones.conv import DeconvBackbone
from old_muzero.configs.modules.backbones.base import BackboneConfig
from old_muzero.configs.modules.backbones.resnet import ResNetConfig
from old_muzero.configs.modules.backbones.denseresnet import DenseResNetConfig
from old_muzero.configs.modules.backbones.dense import DenseConfig
from old_muzero.configs.modules.backbones.conv import ConvConfig
from old_muzero.configs.modules.backbones.recurrent import RecurrentConfig
from old_muzero.configs.modules.backbones.transformer import TransformerConfig
from old_muzero.configs.modules.backbones.identity import IdentityConfig
from old_muzero.configs.modules.backbones.deconv import DeconvConfig


class BackboneFactory:
    """Factory to create Backbone modules based on their configuration."""

    _backbones: Dict[Type[BackboneConfig], Type[nn.Module]] = {
        ResNetConfig: ResNetBackbone,
        DenseResNetConfig: DenseResNetBackbone,
        DenseConfig: DenseBackbone,
        ConvConfig: ConvBackbone,
        RecurrentConfig: RecurrentBackbone,
        TransformerConfig: TransformerBackbone,
        IdentityConfig: IdentityBackbone,
        DeconvConfig: DeconvBackbone,
    }

    @classmethod
    def create(
        cls, config: Optional[BackboneConfig], input_shape: Tuple[int, ...]
    ) -> nn.Module:
        if config is None:
            return IdentityBackbone(None, input_shape)
        config_type = type(config)
        if config_type not in cls._backbones:
            raise ValueError(
                f"No backbone module registered for config type: {config_type}"
            )

        return cls._backbones[config_type](config, input_shape)
