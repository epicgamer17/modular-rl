from typing import Tuple, Optional, Dict, Any
import torch
from torch import Tensor
from .base import BaseHead, HeadOutput
from agents.learner.losses.representations import (
    BaseRepresentation,
    ClassificationRepresentation,
)
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig
from modules.backbones.factory import BackboneFactory
from modules.backbones.mlp import build_dense, NoisyLinear


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
        name: Optional[str] = None,
        input_source: str = "default",
    ):
        if representation is None:
            representation = ClassificationRepresentation(num_classes=num_players)
        super().__init__(arch_config, input_shape, representation, neck_config, name=name, input_source=input_source)

        # 1. Heads now build their own feature architecture (neck)
        self.neck = BackboneFactory.create(neck_config, input_shape)
        self.output_shape = self.neck.output_shape
        self.flat_dim = self._get_flat_dim(self.output_shape)

        # 2. Heads now define their own Final Output layer
        self.output_layer = build_dense(
            in_features=self.flat_dim,
            out_features=self.representation.num_features,
            sigma=self.arch_config.noisy_sigma,
        )

    def reset_noise(self) -> None:
        """Propagate noise reset through the head's submodules."""
        if hasattr(self.neck, "reset_noise"):
            self.neck.reset_noise()
        if isinstance(self.output_layer, NoisyLinear):
            self.output_layer.reset_noise()

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> HeadOutput:
        """Returns HeadOutput with (logits, player_idx, state)"""
        # 1. Processing neck -> flatten
        x = self.neck(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)

        # 2. Final Output Projection
        logits = self.output_layer(x)

        # 3. Mathematical Transform
        player_idx = self.representation.to_expected_value(logits).long()

        return HeadOutput(
            training_tensor=logits,
            inference_tensor=player_idx,
            state=state if state is not None else {},
        )


class RelativeToPlayHead(ToPlayHead):
    """
    Predicts the relative turn shift (ΔP) instead of absolute player index.
    The logits represent probabilities for ΔP ∈ {0, 1, ..., num_players-1}.
    """

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> HeadOutput:
        """Returns HeadOutput with (logits, next_player_idx, state)"""
        # 1. Get raw logits from ToPlayHead's projection
        res = super().forward(x, state)
        logits = res.training_tensor

        # 2. Extract current player index from state
        if state is None:
            state = {}
        current_player_idx = state.get(
            f"{self.name}_current_player_idx",
            torch.zeros(x.shape[0], device=x.device, dtype=torch.long),
        )

        # 3. Calculate the shift (ΔP)
        delta_p = self.representation.to_expected_value(logits).long()

        # 4. Calculate actual next player index: (current + shift) % num_players
        num_players = self.representation.num_features
        player_idx = (current_player_idx + delta_p) % num_players

        # 5. Update the opaque state
        new_state = state.copy()
        new_state[f"{self.name}_current_player_idx"] = player_idx

        return HeadOutput(
            training_tensor=logits,
            inference_tensor=player_idx,
            state=new_state,
        )
