from typing import Tuple, Optional, Dict, Any
from torch import nn, Tensor
from .base import BaseHead
from learner.losses.representations import BaseRepresentation


class ValueHead(BaseHead):
    """
    Predicts the expected return (Value).
    Supports multiple output strategies (Scalar, MuZero, C51, Dreamer).
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
    ) -> Tuple[Tensor, Dict[str, Any], Tensor]:
        """Returns: (logits, state, expected_value)"""
        logits, new_state = super().forward(x, state)
        expected_value = self.representation.to_expected_value(logits)
        return logits, new_state, expected_value
