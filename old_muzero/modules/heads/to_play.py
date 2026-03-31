from typing import Tuple, Optional, Dict, Any
from torch import Tensor
import torch
from .base import BaseHead
from old_muzero.agents.learner.losses.representations import BaseRepresentation, ClassificationRepresentation
from old_muzero.configs.modules.architecture_config import ArchitectureConfig
from old_muzero.configs.modules.backbones.base import BackboneConfig


class ToPlayHead(BaseHead):
    """
    Predicts which player is currently active.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        num_players: int,
        representation: Optional[BaseRepresentation] = None,
        neck_config: Optional[BackboneConfig] = None,
    ):
        if representation is None:
            representation = ClassificationRepresentation(num_classes=num_players)
        super().__init__(arch_config, input_shape, representation, neck_config)

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Any]], Tensor]:
        """Returns: (logits, state, player_idx)"""
        logits, new_state = super().forward(x, state)
        player_idx = self.representation.to_expected_value(logits).long()
        return logits, new_state, player_idx


class RelativeToPlayHead(ToPlayHead):
    """
    Predicts the relative turn shift (ΔP) instead of absolute player index.
    The logits represent probabilities for ΔP ∈ {0, 1, ..., num_players-1}.
    """

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any], Tensor]:
        """Returns: (logits, state, player_idx)"""
        # 1. Get logits from BaseHead
        logits, _ = super(ToPlayHead, self).forward(x, state)

        # 2. Extract current player index from state
        if state is None:
            state = {}
        current_player_idx = state.get(
            "current_player_idx",
            torch.zeros(x.shape[0], device=x.device, dtype=torch.long),
        )

        # 3. Calculate the shift (ΔP)
        delta_p = self.representation.to_expected_value(logits).long()

        # 4. Calculate actual next player index: (current + shift) % num_players
        num_players = self.representation.num_features
        player_idx = (current_player_idx + delta_p) % num_players

        # 5. Update the opaque state
        new_state = state.copy()
        new_state["current_player_idx"] = player_idx

        return logits, new_state, player_idx
