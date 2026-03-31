from typing import Type, Dict, Tuple, Optional
import torch
from torch import nn
from modules.backbones.resnet import ResNetBackbone
from modules.backbones.mlp_resnet import MLPResNetBackbone
from modules.backbones.mlp import MLPBackbone
from modules.backbones.conv import ConvBackbone
from modules.backbones.recurrent import RecurrentBackbone
from modules.backbones.transformer import TransformerBackbone
from modules.backbones.identity import IdentityBackbone
from modules.backbones.conv import DeconvBackbone
from configs.modules.backbones.base import BackboneConfig
from configs.modules.backbones.resnet import ResNetConfig
from configs.modules.backbones.mlp_resnet import MLPResNetConfig
from configs.modules.backbones.mlp import MLPConfig
from configs.modules.backbones.conv import ConvConfig
from configs.modules.backbones.recurrent import RecurrentConfig
from configs.modules.backbones.transformer import TransformerConfig
from configs.modules.backbones.identity import IdentityConfig
from configs.modules.backbones.deconv import DeconvConfig


class BackboneFactory:
    """Factory to create Backbone modules based on their configuration."""

    _backbones: Dict[Type[BackboneConfig], Type[nn.Module]] = {
        ResNetConfig: ResNetBackbone,
        MLPResNetConfig: MLPResNetBackbone,
        MLPConfig: MLPBackbone,
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
            return IdentityBackbone(input_shape)

        config_type = type(config)
        if config_type not in cls._backbones:
            raise ValueError(
                f"No backbone module registered for config type: {config_type}"
            )

        if isinstance(config, MLPConfig):
            return MLPBackbone(
                input_shape=input_shape,
                widths=config.widths,
                activation=config.activation,
                noisy_sigma=config.noisy_sigma,
                norm_type=config.norm_type,
            )

        if isinstance(config, MLPResNetConfig):
            return MLPResNetBackbone(
                input_shape=input_shape,
                widths=config.widths,
                activation=config.activation,
                norm_type=config.norm_type,
                noisy_sigma=config.noisy_sigma,
            )

        if isinstance(config, ConvConfig):
            return ConvBackbone(
                input_shape=input_shape,
                filters=config.filters,
                kernel_sizes=config.kernel_sizes,
                strides=config.strides,
                activation=config.activation,
                noisy_sigma=config.noisy_sigma,
                norm_type=config.norm_type,
            )

        if isinstance(config, DeconvConfig):
            return DeconvBackbone(
                input_shape=input_shape,
                filters=config.filters,
                kernel_sizes=config.kernel_sizes,
                strides=config.strides,
                activation=config.activation,
                norm_type=config.norm_type,
                output_padding=config.output_padding,
            )

        if isinstance(config, ResNetConfig):
            return ResNetBackbone(
                input_shape=input_shape,
                filters=config.filters,
                kernel_sizes=config.kernel_sizes,
                strides=config.strides,
                activation=config.activation,
                noisy_sigma=config.noisy_sigma,
                norm_type=config.norm_type,
            )

        if isinstance(config, RecurrentConfig):
            return RecurrentBackbone(
                input_shape=input_shape,
                rnn_type=config.rnn_type,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
            )

        if isinstance(config, TransformerConfig):
            return TransformerBackbone(
                input_shape=input_shape,
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                d_ff=config.d_ff,
                dropout=config.dropout,
            )

        if isinstance(config, IdentityConfig):
            return IdentityBackbone(input_shape=input_shape)

        raise ValueError(f"Unsupported backbone config type: {config_type}")
