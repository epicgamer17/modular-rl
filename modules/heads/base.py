from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import torch
from torch import nn, Tensor
from modules.utils import get_flat_dim
from configs.modules.backbones.base import BackboneConfig
from configs.modules.architecture_config import ArchitectureConfig
from agents.learner.losses.representations import BaseRepresentation
from abc import ABC, abstractmethod


@dataclass
class HeadOutput:
    """Strict contract for head outputs."""

    training_tensor: torch.Tensor  # e.g., logits, pre-tanh values (for the Learner)
    inference_tensor: Any  # e.g., td.Distribution, argmax action, softmaxed probs (for the Actor)
    state: Dict[str, torch.Tensor] = field(default_factory=dict)  # For recurrent heads
    metrics: Dict[str, float] = field(default_factory=dict)  # Stateless telemetry from the head


class BaseHead(nn.Module, ABC):
    """
    Abstract Base Class for all network heads.
    Enforces the initialization signature and the forward contract.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        neck_config: Optional[BackboneConfig] = None,
        name: Optional[str] = None,
        input_source: str = "default",
    ):
        super().__init__()
        self.arch_config = arch_config
        self.input_shape = input_shape
        self.representation = representation
        self.name = name or self.__class__.__name__
        self.input_source = input_source

    def _get_flat_dim(self, module: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """Utility for heads to calculate their output feature dimension using a dummy pass."""
        return get_flat_dim(module, input_shape)

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
        is_inference: bool = False,
        **kwargs,
    ) -> HeadOutput:
        """Returns HeadOutput conforming to the (training, inference, state) contract."""
        pass

    def compute_metrics(
        self,
        training_tensor: torch.Tensor,
        inference_tensor: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Stateless reporting of head-specific diagnostics.
        Override this to provide telemetry (e.g., entropy, mean value).
        """
        return {}
