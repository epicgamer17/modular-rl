import torch
import torch.nn.functional as F
from typing import Any, Optional, Dict, Union, List
from core import PipelineComponent
from core import Blackboard
from core.path_resolver import resolve_blackboard_path
from .infrastructure import apply_infrastructure


class QBootstrappingLoss(PipelineComponent):
    """Standard TD target loss for Q-learning."""

    def __init__(
        self,
        is_categorical: bool = False,
        loss_fn: Any = None,
        mask_key: str = "value_mask",
        actions_key: str = "actions",
        target_key: Optional[str] = None,
        name: str = "q_loss",
    ):
        self.is_categorical = is_categorical
        self.pred_key = "q_logits" if is_categorical else "q_values"
        
        if target_key:
            self.target_key = target_key
        else:
            self.target_key = "q_logits" if is_categorical else "values"

        if loss_fn is None:
            self.loss_fn = F.cross_entropy if is_categorical else F.mse_loss
        else:
            self.loss_fn = loss_fn

        self.mask_key = mask_key
        self.actions_key = actions_key
        self.name = name

    @property
    def reads(self) -> set[str]:
        return {f"predictions.{self.pred_key}", self.actions_key, self.target_key}

    @property
    def writes(self) -> set[str]:
        return {f"losses.{self.name}", f"meta.{self.name}"}

    def execute(self, blackboard: Blackboard) -> None:
        q_preds = blackboard.predictions[self.pred_key]
        actions = resolve_blackboard_path(blackboard, self.actions_key).long()
        formatted_target = resolve_blackboard_path(blackboard, self.target_key)

        B, T = actions.shape[:2]
        num_actions = q_preds.shape[2]

        flat_preds = q_preds.reshape(B * T, num_actions, -1)
        flat_actions = actions.reshape(-1)
        selected_preds = flat_preds[
            torch.arange(B * T, device=q_preds.device), flat_actions
        ]

        flat_targets = formatted_target.reshape(B * T, -1)

        if selected_preds.ndim > 1 and selected_preds.shape[-1] == 1:
            selected_preds = selected_preds.squeeze(-1)
        if flat_targets.ndim > 1 and flat_targets.shape[-1] == 1:
            flat_targets = flat_targets.squeeze(-1)

        if self.pred_key == "q_logits":
            log_probs = F.log_softmax(selected_preds, dim=-1)
            raw_loss = -(flat_targets * log_probs).sum(dim=-1)
        else:
            raw_loss = self.loss_fn(selected_preds, flat_targets, reduction="none")

        elementwise_loss = raw_loss.reshape(B, T)

        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()

        if "elementwise_losses" not in blackboard.meta:
            blackboard.meta["elementwise_losses"] = {}
        blackboard.meta["elementwise_losses"][self.name] = elementwise_loss
