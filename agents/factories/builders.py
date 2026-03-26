from typing import Callable, Optional, Tuple, Dict, Any, Type
import torch
from torch import nn
from functools import partial

# Backbones
from modules.backbones.resnet import ResNetBackbone
from modules.backbones.mlpresnet import MLPResNetBackbone
from modules.backbones.mlp import MLPBackbone
from modules.backbones.conv import ConvBackbone, DeconvBackbone
from modules.backbones.recurrent import RecurrentBackbone
from modules.backbones.transformer import TransformerBackbone
from modules.backbones.identity import IdentityBackbone

# Heads
from modules.heads.to_play import ToPlayHead
from modules.heads.reward import RewardHead, ValuePrefixRewardHead
from modules.heads.value import ValueHead
from modules.heads.policy import PolicyHead
from modules.heads.chance_probability import ChanceProbabilityHead
from modules.heads.continuation import ContinuationHead
from modules.heads.observation import ObservationHead
from modules.heads.latent_consistency import SimSiamProjectorHead
from modules.heads.q import QHead, DuelingQHead

# Configs (for mapping)
from configs.modules.backbones.base import BackboneConfig
from configs.modules.backbones.resnet import ResNetConfig
from configs.modules.backbones.mlpresnet import MLPResNetConfig
from configs.modules.backbones.mlp import MLPConfig
from configs.modules.backbones.conv import ConvConfig
from configs.modules.backbones.deconv import DeconvConfig
from configs.modules.backbones.recurrent import RecurrentConfig
from configs.modules.backbones.transformer import TransformerConfig
from configs.modules.backbones.identity import IdentityConfig

from configs.modules.heads.base import HeadConfig
from configs.modules.heads.to_play import ToPlayHeadConfig
from configs.modules.heads.value import ValueHeadConfig
from configs.modules.heads.reward import RewardHeadConfig, ValuePrefixRewardHeadConfig
from configs.modules.heads.policy import PolicyHeadConfig
from configs.modules.heads.chance_probability import ChanceProbabilityHeadConfig
from configs.modules.heads.continuation import ContinuationHeadConfig
from configs.modules.heads.observation import ObservationHeadConfig
from configs.modules.heads.latent_consistency import SimSiamProjectorConfig
from configs.modules.heads.q import QHeadConfig, DuelingQHeadConfig

# Mapping of config types to module classes
BACKBONE_MAPPING: Dict[Type[BackboneConfig], Type[nn.Module]] = {
    ResNetConfig: ResNetBackbone,
    MLPResNetConfig: MLPResNetBackbone,
    MLPConfig: MLPBackbone,
    ConvConfig: ConvBackbone,
    RecurrentConfig: RecurrentBackbone,
    TransformerConfig: TransformerBackbone,
    IdentityConfig: IdentityBackbone,
    DeconvConfig: DeconvBackbone,
}

HEAD_MAPPING: Dict[Type[HeadConfig], Type[nn.Module]] = {
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

def make_backbone_fn(config: Optional[BackboneConfig]) -> Optional[Callable]:
    """Returns a partial that instantiates a Backbone module."""
    if config is None:
        return None
    
    config_type = type(config)
    if config_type not in BACKBONE_MAPPING:
        raise ValueError(f"No backbone module registered for config type: {config_type}")
    
    cls = BACKBONE_MAPPING[config_type]
    
    # Extract parameters from config object
    exclude = {"config_dict", "game", "_parsed_fields", "agent_type"}
    kwargs = {k: v for k, v in vars(config).items() if k not in exclude}
    
    return partial(cls, **kwargs)

def make_head_fn(config: HeadConfig, **kwargs) -> Callable:
    """Returns a partial that instantiates a Head module."""
    config_type = type(config)
    if config_type not in HEAD_MAPPING:
        raise ValueError(f"No head module registered for config type: {config_type}")
    
    head_cls = HEAD_MAPPING[config_type]
    
    # 1. Resolve representation if requested
    representation = kwargs.get("representation")
    if (
        representation is None
        and hasattr(config, "output_strategy")
        and config.output_strategy is not None
    ):
        from agents.learner.losses.representations import get_representation
        representation = get_representation(config.output_strategy)
    
    # 2. Resolve neck_fn if present
    # Heads now take neck_fn instead of neck_config.
    # We convert the neck_config into a neck_fn here.
    neck_fn = None
    if hasattr(config, "neck") and config.neck is not None:
        neck_fn = make_backbone_fn(config.neck)
    
    # 3. Build partial with combined arguments
    # We combine config fields with explicit kwargs (like num_players, num_actions)
    exclude = {"config_dict", "game", "_parsed_fields", "agent_type", "neck", "output_strategy"}
    head_kwargs = {k: v for k, v in vars(config).items() if k not in exclude}
    head_kwargs.update(kwargs)
    
    # Inject resolved dependencies
    head_kwargs["representation"] = representation
    head_kwargs["neck_fn"] = neck_fn
    
    return partial(head_cls, **head_kwargs)
