from typing import Type, Dict, Tuple, Optional
import torch
from torch import nn
from modules.backbones.resnet import ResNetBackbone
from modules.backbones.denseresnet import DenseResNetBackbone
from modules.backbones.dense import DenseBackbone
from modules.backbones.conv import ConvBackbone
from modules.backbones.recurrent import RecurrentBackbone
from modules.backbones.transformer import TransformerBackbone
from modules.backbones.identity import IdentityBackbone
from modules.backbones.conv import DeconvBackbone
from configs.modules.backbones.base import BackboneConfig
from configs.modules.backbones.resnet import ResNetConfig
from configs.modules.backbones.denseresnet import DenseResNetConfig
from configs.modules.backbones.dense import DenseConfig
from configs.modules.backbones.conv import ConvConfig
from configs.modules.backbones.recurrent import RecurrentConfig
from configs.modules.backbones.transformer import TransformerConfig
from configs.modules.backbones.identity import IdentityConfig
from configs.modules.backbones.deconv import DeconvConfig


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
