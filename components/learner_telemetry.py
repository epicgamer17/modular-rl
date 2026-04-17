import torch
from typing import Any, Tuple, Dict, Set
from core import PipelineComponent, Blackboard
from core.path_resolver import resolve_blackboard_path

from core.contracts import Key, ToPlay, ValueEstimate, ValueTarget, LossScalar, Metric

class MuzeroMultiplayerTelemetry(PipelineComponent):
    """
    Learner telemetry component to track performance metrics split by player (0 vs 1).
    """
    required = True


    def __init__(
        self,
        value_representation: Any = None,
        num_players: int = 2,
        to_play_pred_key: str = "to_plays",
        to_play_target_key: str = "to_plays",
        value_pred_key: str = "values",
        value_target_key: str = "values",
        mask_key: str = "masks",
    ):
        self.value_representation = value_representation
        self.num_players = num_players
        self.to_play_pred_key = to_play_pred_key
        self.to_play_target_key = to_play_target_key
        self.value_pred_key = value_pred_key
        self.value_target_key = value_target_key
        self.mask_key = mask_key

        # Deterministic contracts computed at initialization
        self._requires = {
            Key(f"predictions.{self.to_play_pred_key}", ToPlay),
            Key(self.to_play_target_key, ToPlay),
            Key(f"predictions.{self.value_pred_key}", ValueEstimate),
            Key(self.value_target_key, ValueTarget),
        }
        self._provides = {}
        for p in range(self.num_players):
            self._provides[Key(f"meta.tp_acc_p{p}", Metric)] = "new"
            self._provides[Key(f"meta.val_mse_p{p}", Metric)] = "new"
            self._provides[Key(f"meta.losses.tp_acc_p{p}", Metric)] = "new"
            self._provides[Key(f"meta.losses.val_mse_p{p}", Metric)] = "new"

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures to-play predictions exist."""
        if self.to_play_pred_key not in blackboard.predictions:
            return  # Component gracefully skips if no predictions
        from core.validation import assert_is_tensor, assert_same_batch
        tp_preds = blackboard.predictions[self.to_play_pred_key]
        assert_is_tensor(tp_preds, msg="in MuzeroMultiplayerTelemetry (to_play predictions)")

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        if self.to_play_pred_key not in blackboard.predictions:
            return {}

        # 1. Extract standard tensors
        tp_preds = blackboard.predictions[self.to_play_pred_key]
        tp_targets = resolve_blackboard_path(blackboard, self.to_play_target_key)
        
        val_preds = blackboard.predictions[self.value_pred_key]
        val_targets = resolve_blackboard_path(blackboard, self.value_target_key)
        
        # Get mask (B, T)
        mask = blackboard.meta.get(self.mask_key)
        if mask is None:
            mask = torch.ones(tp_preds.shape[:2], device=tp_preds.device)

        # 2. ToPlay Accuracy
        # tp_targets is one-hot [B, T, C] or indices [B, T]
        if tp_targets.ndim == 3:
            target_ids = tp_targets.argmax(dim=-1)
        else:
            target_ids = tp_targets.long()
            
        pred_ids = tp_preds.argmax(dim=-1)
        correct = (pred_ids == target_ids).float()
        
        # 3. Value Error
        # Convert logits to expected value if representation is provided
        if self.value_representation is not None:
            val_scalars = self.value_representation.to_expected_value(val_preds)
        else:
            val_scalars = val_preds
            if val_scalars.ndim == 3:
                val_scalars = val_scalars.squeeze(-1)

        if val_targets.ndim == 3:
            val_targets = val_targets.squeeze(-1)
            
        val_error = (val_scalars - val_targets)**2
        
        updates = {}
        # 4. Split and Log
        for p in range(self.num_players):
            p_mask = (target_ids == p) & mask.bool()
            count = p_mask.sum().item()
            
            if count > 0:
                acc = (correct[p_mask]).mean().item()
                mse = (val_error[p_mask]).mean().item()
                
                updates[f"meta.tp_acc_p{p}"] = acc
                updates[f"meta.val_mse_p{p}"] = mse
                
                # Also expose directly to learner metrics for printing
                updates[f"meta.losses.tp_acc_p{p}"] = acc
                updates[f"meta.losses.val_mse_p{p}"] = mse

        return updates
