from typing import Callable, Tuple
from torch import Tensor
from configs.agents.supervised import SupervisedConfig
from modules.backbones.factory import BackboneFactory
from modules.blocks.dense import build_dense
import torch.nn as nn

# TODO: UPDATE THIS TO THE BaseAgentNetwork


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

    def initial_inference(self, inputs: Tensor) -> "InferenceOutput":
        from modules.world_models.inference_output import InferenceOutput
        from torch.distributions import Categorical

        # Ensure inputs have batch dimension
        if inputs.dim() == len(self.config.backbone.input_shape):
            inputs = inputs.unsqueeze(0)

        x = self.backbone(inputs)

        if x.dim() > 2:
            x = x.flatten(1, -1)

        x: Tensor = self.output_layer(x).view(-1, self.output_size)

        logits = x

        if self.return_logits:
            # If return_logits is True, we construct Categorical from logits
            dist = Categorical(logits=logits)
        else:
            # If previously it returned softmax, now we handle it.
            # But 'x' is just output of dense layer (logits usually).
            # Previous forward() did x.softmax(dim=-1) if not return_logits.
            # So if we want Categorical, passing logits is stable.
            # We should ignore self.return_logits for Categorical construction?
            # Or if return_logits was meant for loss computation?
            # For InferenceOutput, we want a Distribution.
            # Categorical(logits=logits) is standard.
            dist = Categorical(logits=logits)

        return InferenceOutput(policy=dist)
        if hasattr(self.backbone, "reset_noise"):
            self.backbone.reset_noise()
        if hasattr(self.output_layer, "reset_noise"):
            self.output_layer.reset_noise()

    def initial_inference(self, inputs: Tensor) -> "InferenceOutput":
        from modules.world_models.inference_output import InferenceOutput
        from torch.distributions import Categorical

        # Ensure inputs have batch dimension
        if inputs.dim() == len(self.config.backbone.input_shape):
            inputs = inputs.unsqueeze(0)

        logits = self.forward(inputs)
        # Check if forward returns logits or probs based on self.return_logits
        # If probs, convert to log_probs for numerical stability or just use probs in Categorical?
        # Categorical can take probs or logits.

        if self.return_logits:
            dist = Categorical(logits=logits)
        else:
            dist = Categorical(probs=logits)

        return InferenceOutput(policy=dist)
