from typing import Type, Dict, Any, Optional


class BackboneConfigFactory:
    """Factory to create BackboneConfig instances based on type."""

    @classmethod
    def create(cls, config_dict: dict) -> Any:
        if config_dict is None:
            return None

        from configs.modules.backbones.resnet import ResNetConfig
        from configs.modules.backbones.deconv import DeconvConfig
        from configs.modules.backbones.mlpresnet import MLPResNetConfig
        from configs.modules.backbones.mlp import MLPConfig
        from configs.modules.backbones.conv import ConvConfig
        from configs.modules.backbones.recurrent import RecurrentConfig
        from configs.modules.backbones.transformer import TransformerConfig
        from configs.modules.backbones.identity import IdentityConfig
        from configs.modules.backbones.base import BackboneConfig

        configs: Dict[str, Type[BackboneConfig]] = {
            "resnet": ResNetConfig,
            "mlpresnet": MLPResNetConfig,
            "mlp": MLPConfig,
            "conv": ConvConfig,
            "recurrent": RecurrentConfig,
            "transformer": TransformerConfig,
            "identity": IdentityConfig,
            "deconv": DeconvConfig,
        }

        if "type" not in config_dict:
            # Default to resnet if filters is present, or dense if widths is present
            if "filters" in config_dict:
                bb_type = "resnet"
            elif "widths" in config_dict:
                bb_type = "mlp"
            else:
                raise ValueError(
                    "Backbone config must contain 'type' or identifiable fields."
                )
        else:
            bb_type = config_dict["type"].lower()

        if bb_type not in configs:
            raise ValueError(f"Unknown backbone type: {bb_type}")

        return configs[bb_type](config_dict)
