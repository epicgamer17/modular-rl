from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, Callable
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
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        neck_fn: Optional[Callable[[Tuple[int, ...]], nn.Module]] = None,
        noisy_sigma: float = 0.0,
        name: Optional[str] = None,
        input_source: str = "default",
        **kwargs,
    ):
        super().__init__()
        self.noisy_sigma = noisy_sigma
        self.input_shape = input_shape
        self.representation = representation
        self.neck_fn = neck_fn
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

    def init_weights(self) -> None:
        """
        Component-owned initialization strategy.
        Base implementation uses orthogonal initialization (gain=1.0) and zero bias.
        """
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
