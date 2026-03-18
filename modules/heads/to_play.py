from typing import Tuple, Optional, Dict, Any
from torch import Tensor
import torch
from .base import BaseHead
from modules.heads.strategies import Categorical
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig


class ToPlayHead(BaseHead):
    """
    Predicts which player is currently active.

    Follows the same 3-tuple return contract as ``RewardHead`` and ``PolicyHead``:

    - ``logits``     – raw pre-softmax logits for the cross-entropy loss.
    - ``new_state``  – opaque head state (``None`` for stateless heads).
    - ``player_idx`` – inferred output for the actor/MCTS: the argmax player index
                       as a scalar integer tensor of shape ``(B,)``.

    The learner uses ``logits`` directly with ``F.cross_entropy``.
    The actor/MCTS uses ``player_idx`` to determine whose turn it is.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        num_players: int,
        neck_config: Optional[BackboneConfig] = None,
        strategy=None,
    ):
        """
        Args:
            arch_config:  Shared architecture config (activation, norm type, etc.).
            input_shape:  Shape of the input feature tensor (excluding batch dim).
            num_players:  Number of players (= number of output classes).
            neck_config:  Optional lightweight neck backbone before the output layer.
            strategy:     Output strategy; defaults to ``Categorical(num_players)``.
        """
        if strategy is None:
            strategy = Categorical(num_classes=num_players)
        super().__init__(arch_config, input_shape, strategy, neck_config)

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Any]], Tensor]:
        """
        Run the to-play head.

        Args:
            x:     Input feature tensor of shape ``(B, *input_shape)``.
            state: Optional opaque head state (unused; kept for interface symmetry).

        Returns:
            logits:     Raw logits of shape ``(B, num_players)`` — used by the loss.
            new_state:  Opaque head state (``None`` for this stateless head).
            player_idx: Argmax player indices of shape ``(B,)`` — used by the actor
                        and MCTS to determine whose turn it is.
        """
        logits, new_state = super().forward(x, state)
        player_idx = self.strategy.to_expected_value(logits).long()  # (B,)
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
        """
        Run the relative to-play head.

        Args:
            x:     Input feature tensor of shape ``(B, *input_shape)``.
            state: Opaque head state containing 'current_player_idx'.

        Returns:
            logits:     Raw logits of shape ``(B, num_players)`` representing ΔP.
            new_state:  Updated head state with the new 'current_player_idx'.
            player_idx: Actual next player index of shape ``(B,)``.
        """
        # 1. Get logits from BaseHead (via super.forward which calls process_input and output_layer)
        # Note: ToPlayHead.forward also calls argmax, but we re-calculate it here for ΔP.
        logits, _ = super(ToPlayHead, self).forward(x, state)

        # 2. Extract current player index from state (Opaque Token)
        if state is None:
            state = {}
        current_player_idx = state.get(
            "current_player_idx",
            torch.zeros(x.shape[0], device=x.device, dtype=torch.long),
        )

        # 3. Calculate the shift (ΔP)
        delta_p = self.strategy.to_expected_value(logits).long()

        # 4. Calculate actual next player index: (current + shift) % num_players
        # self.strategy.num_bins is defined in Categorical strategy (initialized in ToPlayHead)
        num_players = self.strategy.num_bins
        player_idx = (current_player_idx + delta_p) % num_players

        # 5. Update the opaque state
        new_state = state.copy()
        new_state["current_player_idx"] = player_idx

        return logits, new_state, player_idx
