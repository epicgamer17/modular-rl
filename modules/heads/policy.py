from typing import Tuple, Optional, Dict, Any
import torch
from torch import nn, Tensor
from .base import BaseHead
from learner.losses.representations import BaseRepresentation


class PolicyHead(BaseHead):
    """
    Predicts the action distribution (Policy).
    Supports both discrete (Categorical) and continuous (Gaussian) actions via Representation.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        neck: Optional[nn.Module] = None,
        noisy_sigma: float = 0.0,
    ):
        super().__init__(input_shape, representation, neck, noisy_sigma)

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any], Any]:
        """Returns: (logits, state, inference)"""
        logits, new_state = super().forward(x, state)
        inference = self.representation.to_inference(logits)
        return logits, new_state, inference
