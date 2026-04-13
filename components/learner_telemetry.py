import torch
from core import PipelineComponent, Blackboard
from core.path_resolver import resolve_blackboard_path

class MuzeroMultiplayerTelemetry(PipelineComponent):
    """
    Learner telemetry component to track performance metrics split by player (0 vs 1).
    """
    def __init__(
        self,
        to_play_pred_key: str = "to_plays",
        to_play_target_key: str = "to_plays",
        value_pred_key: str = "values",
        value_target_key: str = "values",
        mask_key: str = "masks",
    ):
        self.to_play_pred_key = to_play_pred_key
        self.to_play_target_key = to_play_target_key
        self.value_pred_key = value_pred_key
        self.value_target_key = value_target_key
        self.mask_key = mask_key

    def execute(self, blackboard: Blackboard) -> None:
        if self.to_play_pred_key not in blackboard.predictions:
            return

        # 1. Extract standard tensors
        tp_preds = blackboard.predictions[self.to_play_pred_key]
        tp_targets = resolve_blackboard_path(blackboard, f"targets.{self.to_play_target_key}")
        
        val_preds = blackboard.predictions[self.value_pred_key]
        val_targets = resolve_blackboard_path(blackboard, f"targets.{self.value_target_key}")
        
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
        # Ensure value shapes match [B, T]
        if val_preds.ndim == 3:
            val_preds = val_preds.squeeze(-1)
        if val_targets.ndim == 3:
            val_targets = val_targets.squeeze(-1)
            
        val_error = (val_preds - val_targets)**2
        
        # 4. Split and Log
        for p in range(2): # 2 players for Tic-Tac-Toe
            p_mask = (target_ids == p) & mask.bool()
            count = p_mask.sum().item()
            
            if count > 0:
                acc = (correct[p_mask]).mean().item()
                mse = (val_error[p_mask]).mean().item()
                
                blackboard.meta[f"tp_acc_p{p}"] = acc
                blackboard.meta[f"val_mse_p{p}"] = mse
                
                # Also expose directly to learner metrics for printing
                if "losses" not in blackboard.meta:
                    blackboard.meta["losses"] = {}
                blackboard.meta["losses"][f"tp_acc_p{p}"] = acc
                blackboard.meta["losses"][f"val_mse_p{p}"] = mse
