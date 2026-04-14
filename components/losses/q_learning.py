import torch
import torch.nn.functional as F
from typing import Any, Optional, Dict, Union, List, Set
from core import PipelineComponent, Blackboard
from core.path_resolver import resolve_blackboard_path
from core.contracts import Key, PolicyLogits, ValueEstimate, Action, ValueTarget, LossScalar, Metric, Scalar, Categorical, Logits
from .infrastructure import apply_infrastructure


class QBootstrappingLoss(PipelineComponent):
    """Standard TD target loss for Q-learning."""

    def __init__(
        self,
        is_categorical: bool = False,
        atom_size: Optional[int] = None,
        loss_fn: Any = None,
        mask_key: str = "value_mask",
        actions_key: str = "actions",
        target_key: Optional[str] = None,
        name: str = "q_loss",
    ):
        self.is_categorical = is_categorical
        self.atom_size = atom_size
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

        # Deterministic contracts computed at initialization
        if self.is_categorical:
            # If atoms are used, we expect Categorical structure
            struct = Categorical(bins=atom_size) if atom_size else Logits()
            req_type = PolicyLogits[struct]
            target_type = PolicyLogits[struct] # Or Probs if target is normalized
        else:
            req_type = ValueEstimate[Scalar]
            target_type = ValueTarget[Scalar]

        self._requires = {
            Key(f"predictions.{self.pred_key}", req_type),
            Key(self.actions_key, Action),
            Key(self.target_key, target_type)
        }
        self._provides = {
            Key(f"losses.{self.name}", LossScalar): "new",
            Key(f"meta.{self.name}", Metric): "new",
            Key(f"meta.elementwise_losses.{self.name}", Metric): "new",
        }

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures predictions, actions, and targets exist and are batch-aligned."""
        from core.validation import assert_in_blackboard, assert_is_tensor, assert_same_batch
        assert_in_blackboard(blackboard, f"predictions.{self.pred_key}")
        assert_in_blackboard(blackboard, self.actions_key)
        assert_in_blackboard(blackboard, self.target_key)

        q_preds = blackboard.predictions[self.pred_key]
        actions = resolve_blackboard_path(blackboard, self.actions_key)
        targets = resolve_blackboard_path(blackboard, self.target_key)

        assert_is_tensor(q_preds, msg=f"in {self.name} (predictions)")
        assert_is_tensor(actions, msg=f"in {self.name} (actions)")
        assert_is_tensor(targets, msg=f"in {self.name} (targets)")

        assert q_preds.shape[:2] == actions.shape[:2], (
            f"Batch/Time mismatch in {self.name}: preds {q_preds.shape[:2]} vs actions {actions.shape[:2]}"
        )
        assert_same_batch(q_preds, targets, msg=f"in {self.name}")

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
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

        return {
            f"losses.{self.name}": scalar_loss,
            f"meta.{self.name}": scalar_loss.item(),
            f"meta.elementwise_losses.{self.name}": elementwise_loss
        }
