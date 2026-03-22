from typing import Tuple, Optional, Dict, Any
import torch
from torch import nn, Tensor

from .base import BaseHead, HeadOutput
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig
from agents.learner.losses.representations import BaseRepresentation
from modules.backbones.factory import BackboneFactory
from modules.backbones.mlp import build_dense, NoisyLinear


class QHead(BaseHead):
    """
    Modular Q-Network Head.
    Structure: [Neck (Optional)] -> [Hidden Backbone] -> [Output Layer] -> [Strategy (Reshape)]
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        hidden_backbone_config: BackboneConfig,
        num_actions: int,
        neck_config: Optional[BackboneConfig] = None,
        name: Optional[str] = None,
        input_source: str = "default",
    ):
        super().__init__(arch_config, input_shape, representation, neck_config, name=name, input_source=input_source)

        # 1. Heads now build their own feature architecture (neck)
        self.neck = BackboneFactory.create(neck_config, input_shape)
        self.output_shape = self.neck.output_shape
        self.flat_dim = self._get_flat_dim(self.neck, input_shape)

        self.num_actions = num_actions
        self.noisy = self.arch_config.noisy_sigma != 0

        # 2. Hidden Layers are now a Backbone!
        self.hidden_layers = BackboneFactory.create(
            hidden_backbone_config, (self.flat_dim,)
        )

        # 3. Final Output Layer
        self.output_layer = build_dense(
            in_features=self.hidden_layers.output_shape[0],
            out_features=self.representation.num_features * self.num_actions,
            sigma=self.arch_config.noisy_sigma,
        )

    def reset_noise(self) -> None:
        """Propagate noise reset through the neck and output layers."""
        if hasattr(self.neck, "reset_noise"):
            self.neck.reset_noise()
        if hasattr(self.hidden_layers, "reset_noise"):
            self.hidden_layers.reset_noise()
        if isinstance(self.output_layer, NoisyLinear):
            self.output_layer.reset_noise()

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
        is_inference: bool = False,
        **kwargs,
    ) -> HeadOutput:
        # 1. Processing neck -> flatten
        x = self.neck(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)

        # 2. Hidden Backbone Phase
        x = self.hidden_layers(x)

        # 3. Output Projection Phase
        logits = self.output_layer(x)
        logits = logits.view(-1, self.num_actions, self.representation.num_features)

        # 4. Standard Return
        new_state = state if state is not None else {}
        inference = None
        if is_inference:
            inference = self.representation.to_inference(logits)

        return HeadOutput(
            training_tensor=logits,
            inference_tensor=inference,
            state=new_state,
        )


class DuelingQHead(BaseHead):
    """
    Modular Dueling Q-Network Head.
    Structure:
        [Neck] -> Split -> [Value Backbone] -> [Value Output]
                        -> [Advantage Backbone] -> [Advantage Output]
               -> Aggregation (Q = V + A - mean(A))
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        value_hidden_backbone_config: BackboneConfig,
        advantage_hidden_backbone_config: BackboneConfig,
        num_actions: int,
        neck_config: Optional[BackboneConfig] = None,
        name: Optional[str] = None,
        input_source: str = "default",
    ):
        super().__init__(arch_config, input_shape, representation, neck_config, name=name, input_source=input_source)

        # 1. Heads now build their own feature architecture (neck)
        self.neck = BackboneFactory.create(neck_config, input_shape)
        self.output_shape = self.neck.output_shape
        self.flat_dim = self._get_flat_dim(self.neck, input_shape)

        self.num_actions = num_actions
        self.noisy = self.arch_config.noisy_sigma != 0

        # 2. Value Stream (Backbone + Output)
        self.value_hidden = BackboneFactory.create(
            value_hidden_backbone_config, (self.flat_dim,)
        )
        self.value_output = build_dense(
            in_features=self.value_hidden.output_shape[0],
            out_features=self.representation.num_features,
            sigma=self.arch_config.noisy_sigma,
        )

        # 3. Advantage Stream (Backbone + Output)
        self.advantage_hidden = BackboneFactory.create(
            advantage_hidden_backbone_config, (self.flat_dim,)
        )
        self.advantage_output = build_dense(
            in_features=self.advantage_hidden.output_shape[0],
            out_features=self.representation.num_features * self.num_actions,
            sigma=self.arch_config.noisy_sigma,
        )

    def reset_noise(self) -> None:
        """Propagate noise reset through the neck and stream streams."""
        if hasattr(self.neck, "reset_noise"):
            self.neck.reset_noise()

        for stream in [self.value_hidden, self.advantage_hidden]:
            if hasattr(stream, "reset_noise"):
                stream.reset_noise()

        for out in [self.value_output, self.advantage_output]:
            if isinstance(out, NoisyLinear):
                out.reset_noise()

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
        is_inference: bool = False,
        **kwargs,
    ) -> HeadOutput:
        # 1. Processing neck -> flatten
        x = self.neck(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)

        # 2. Parallel Stream Processing
        v = self.value_hidden(x)
        v = self.value_output(v)  # (B, atoms)
        v = v.view(-1, 1, self.representation.num_features)  # (B, 1, atoms)

        a = self.advantage_hidden(x)
        a = self.advantage_output(a)  # (B, actions * atoms)
        a = a.view(-1, self.num_actions, self.representation.num_features)

        # Aggregation Phase: Q = V + (A - mean(A))
        a_mean = a.mean(dim=1, keepdim=True)
        q = v + a - a_mean

        new_state = state if state is not None else {}
        inference = None
        if is_inference:
            inference = self.representation.to_inference(q)

        return HeadOutput(
            training_tensor=q,
            inference_tensor=inference,
            state=new_state,
        )
