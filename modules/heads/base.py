from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable, Dict, Any
import torch
from torch import nn, Tensor
from modules.backbones.factory import BackboneFactory
from configs.modules.backbones.base import BackboneConfig
from configs.modules.architecture_config import ArchitectureConfig
from agents.learner.losses.representations import BaseRepresentation
from modules.backbones.mlp import build_dense


@dataclass
class HeadOutput:
    """Strict contract for head outputs."""

    training_tensor: torch.Tensor  # e.g., logits, pre-tanh values (for the Learner)
    inference_tensor: Any  # e.g., td.Distribution, argmax action, softmaxed probs (for the Actor)
    state: Dict[str, torch.Tensor] = field(default_factory=dict)  # For recurrent heads


class BaseHead(nn.Module):
    """
    Base class for all network heads.
    Handles an optional neck (modular backbone) and standard initialization.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        neck_config: Optional[BackboneConfig] = None,
    ):
        super().__init__()
        self.arch_config = arch_config
        self.input_shape = input_shape
        self.representation = representation

        # 1. Neck (optional modular backbone associated with the head)
        self.neck = BackboneFactory.create(neck_config, input_shape)
        self.output_shape = self.neck.output_shape
        self.flat_dim = self._get_flat_dim(self.output_shape)

        # 2. Final Output Layer
        self.output_layer = None
        if self.representation is not None:
            self.output_layer = build_dense(
                in_features=self.flat_dim,
                out_features=self.representation.num_features,
                sigma=self.arch_config.noisy_sigma,
            )

    def _get_flat_dim(self, shape: Tuple[int, ...]) -> int:
        flat = 1
        for dim in shape:
            flat *= dim
        return flat

    def reset_noise(self) -> None:
        if hasattr(self.neck, "reset_noise"):
            self.neck.reset_noise()
        if self.output_layer is not None and hasattr(self.output_layer, "reset_noise"):
            self.output_layer.reset_noise()

    def process_input(self, x: Tensor) -> Tensor:
        """Standard input processing: neck -> flatten."""
        x = self.neck(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)
        return x

    def forward(self, x: Tensor, state: Optional[Dict[str, Any]] = None) -> HeadOutput:
        """Standard forward pass: neck -> flatten -> output_layer."""
        x = self.process_input(x)
        logits = self.output_layer(x)
        return HeadOutput(
            training_tensor=logits,
            inference_tensor=logits,
            state=state if state is not None else {},
        )
