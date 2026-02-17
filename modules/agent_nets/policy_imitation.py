from typing import Callable, Tuple
from torch import Tensor
from configs.agents.supervised import SupervisedConfig
from modules.backbones.factory import BackboneFactory
from modules.blocks.dense import build_dense
import torch.nn as nn


class SupervisedNetwork(nn.Module):
    def __init__(
        self,
        config: SupervisedConfig,
        output_size: int,
        input_shape: Tuple[int],
        *args,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.output_size = output_size

        # Core Backbone
        self.backbone = BackboneFactory.create(config.backbone, input_shape)

        input_width = self._get_flat_dim(self.backbone.output_shape)

        self.output_layer = build_dense(
            input_width,
            output_size,
            sigma=self.config.noisy_sigma,
        )
        self.return_logits = getattr(config, "return_logits", False)

    def _get_flat_dim(self, shape: Tuple[int]) -> int:
        flat = 1
        for dim in shape:
            flat *= dim
        return flat

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        self.backbone.initialize(initializer)
        self.output_layer.initialize(initializer)

    def forward(self, inputs: Tensor):
        x = self.backbone(inputs)

        if x.dim() > 2:
            x = x.flatten(1, -1)

        x: Tensor = self.output_layer(x).view(-1, self.output_size)

        if self.return_logits:
            return x
        return x.softmax(dim=-1)

    def reset_noise(self):
        if hasattr(self.backbone, "reset_noise"):
            self.backbone.reset_noise()
        if hasattr(self.output_layer, "reset_noise"):
            self.output_layer.reset_noise()
