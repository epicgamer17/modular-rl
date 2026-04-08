import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional
from learner.pipeline.base import PipelineComponent
from learner.core import Blackboard
from learner.losses.base import apply_infrastructure


class QBootstrappingLoss(PipelineComponent):
    """
    Standard TD target loss for Q-learning.
    Indexes the prediction tensor by the taken actions to compute TD errors.
    """
    def __init__(
        self,
        is_categorical: bool = False,
        loss_fn: Any = None,
        mask_key: str = "value_mask",
        name: str = "q_loss",
    ):
        self.is_categorical = is_categorical
        self.pred_key = "q_logits" if is_categorical else "q_values"
        self.target_key = "q_logits" if is_categorical else "values"
        
        if loss_fn is None:
            self.loss_fn = F.cross_entropy if is_categorical else F.mse_loss
        else:
            self.loss_fn = loss_fn
            
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        q_preds = blackboard.predictions[self.pred_key]
        actions = blackboard.targets["actions"].long()
        formatted_target = blackboard.targets[self.target_key]

        B, T = actions.shape[:2]
        num_actions = q_preds.shape[2]

        # Select Take Action Predictions
        flat_preds = q_preds.reshape(B * T, num_actions, -1)
        flat_actions = actions.reshape(-1)
        selected_preds = flat_preds[torch.arange(B * T, device=q_preds.device), flat_actions]

        # Use pre-formatted Targets
        flat_targets = formatted_target.reshape(B * T, -1)

        # Matching shapes
        if selected_preds.ndim > 1 and selected_preds.shape[-1] == 1:
            selected_preds = selected_preds.squeeze(-1)
        if flat_targets.ndim > 1 and flat_targets.shape[-1] == 1:
            flat_targets = flat_targets.squeeze(-1)

        # Apply Loss Function
        if self.pred_key == "q_logits":
            # Multi-atom categorical cross-entropy
            log_probs = F.log_softmax(selected_preds, dim=-1)
            raw_loss = -(flat_targets * log_probs).sum(dim=-1)
        else:
            # Standard scalar regression (MSE)
            raw_loss = self.loss_fn(selected_preds, flat_targets, reduction="none")

        elementwise_loss = raw_loss.reshape(B, T)

        # Pass through infrastructure
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        # Write out
        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()
        
        # Store elementwise loss for priority computation
        if "elementwise_losses" not in blackboard.meta:
            blackboard.meta["elementwise_losses"] = {}
        blackboard.meta["elementwise_losses"][self.name] = elementwise_loss



class ChanceQLoss(PipelineComponent):
    """Loss for stochastic muzero chance Q heads."""
    def __init__(
        self,
        loss_factor: float = 1.0,
        mask_key: str = "afterstate_value_mask",
        name: str = "chance_q_loss",
    ):
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        formatted_target = blackboard.targets.get("chance_values_next")
        if formatted_target is None:
            raise KeyError("ChanceQLoss requires 'chance_values_next' in targets.")

        pred = blackboard.predictions["chance_q_logits"]
        B, T = pred.shape[:2]

        flat_pred = pred.reshape(B * T, -1)
        flat_target = formatted_target.reshape(B * T, -1)

        raw_loss = F.cross_entropy(flat_pred, flat_target, reduction="none")
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)

        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor

        # Pass through infrastructure
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        # Write out
        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()

