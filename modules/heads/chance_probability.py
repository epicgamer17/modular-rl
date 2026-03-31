from typing import Tuple, Optional, Dict, Any
from torch import nn, Tensor
from .base import BaseHead
from agents.learner.losses.representations import ClassificationRepresentation


class ChanceProbabilityHead(BaseHead):
    """
    Predicts the probability distribution over chance outcomes (codes).
    Used in Stochastic MuZero.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_chance_codes: int,
        neck: Optional[nn.Module] = None,
        noisy_sigma: float = 0.0,
    ):
        representation = ClassificationRepresentation(num_classes=num_chance_codes)
        super().__init__(input_shape, representation, neck, noisy_sigma)

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        logits, new_state = super().forward(x, state)
        inference = self.representation.to_inference(logits)
        return logits, new_state, inference
