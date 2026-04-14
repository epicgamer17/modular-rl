import torch
import torch.nn.functional as F
from typing import Any, Set
from core import PipelineComponent
from core import Blackboard
from core.path_resolver import resolve_blackboard_path
from core.contracts import Key, ValueEstimate, ValueTarget, LossScalar
from .infrastructure import apply_infrastructure


class ValueLoss(PipelineComponent):
    """
    Standard Value prediction loss component.
    Reads 'values' from predictions and targets.
    """

    def __init__(
        self,
        target_key: str = "values",
        mask_key: str = "value_mask",
        loss_fn: Any = F.mse_loss,
        loss_factor: float = 1.0,
        name: str = "value_loss",
    ):
        self.target_key = target_key
        self.mask_key = mask_key
        self.loss_fn = loss_fn
        self.loss_factor = loss_factor
        self.name = name
        
        # Deterministic contracts computed at initialization
        self._requires = {
            Key("predictions.values", ValueEstimate),
            Key(self.target_key, ValueTarget)
        }
        self._provides = {Key(f"losses.{self.name}", LossScalar)}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    @property
    def constraints(self) -> list[str]:
        return [
            f"same_batch(predictions.values, {self.target_key})",
            "time_aligned(predictions.values, targets)"
        ]

    def validate(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions["values"]
        targets = resolve_blackboard_path(blackboard, self.target_key)
        assert preds.shape[0] == targets.shape[0], f"Batch size mismatch: {preds.shape[0]} vs {targets.shape[0]}"
        # Semantic check: Target must be scalar if we are using MSE without atom logic
        assert targets.ndim <= 3, "ValueLoss expects scalar or simple vector targets."

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions["values"]
        targets = resolve_blackboard_path(blackboard, self.target_key)
        B, T = preds.shape[:2]

        # 1. Align targets to predictions feature dimension if missing
        if targets.ndim == preds.ndim - 1:
            targets = targets.unsqueeze(-1)

        # 2. Ensure feature dimension matches if preds is [B, T, 1] and targets is [B, T]
        if preds.ndim == 3 and targets.ndim == 2:
            targets = targets.unsqueeze(-1)

        # 3. Robust squeeze to [N] for both if they are [N, 1] vs [N]
        # or just ensure they match for F.mse_loss (which doesn't like [N, 1] vs [N])
        p_flat = preds.flatten(0, 1)
        t_flat = targets.flatten(0, 1)
        # TODO: enforce always time dimension some how.
        if p_flat.ndim == 2 and p_flat.shape[1] == 1 and t_flat.ndim == 1:
            p_flat = p_flat.squeeze(-1)

        # 3. Flatten B, T for loss function
        raw_loss = self.loss_fn(p_flat, t_flat, reduction="none")

        # Reshape to [B, T]
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)
        B, T = preds.shape[:2]
        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor

        # Pass through infrastructure
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        # Write out
        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()

        # Store elementwise loss for priority computation
        if "elementwise_losses" not in blackboard.meta:
            blackboard.meta["elementwise_losses"] = {}
        blackboard.meta["elementwise_losses"][self.name] = elementwise_loss


class ClippedValueLoss(PipelineComponent):
    """
    PPO Clipped Value Loss.
    Formula: max[(V - V_targ)^2, (clip(V, V_old - eps, V_old + eps) - V_targ)^2]
    """

    def __init__(
        self,
        clip_param: float,
        target_key: str = "returns",
        old_values_key: str = "values",
        mask_key: str = "value_mask",
        loss_factor: float = 1.0,
        name: str = "value_loss",
    ):
        self.clip_param = clip_param
        self.target_key = target_key
        self.old_values_key = old_values_key
        self.mask_key = mask_key
        self.loss_factor = loss_factor
        self.name = name

    @property
    def requires(self) -> set[Key]:
        return {
            Key("predictions.values", ValueEstimate),
            Key(self.target_key, ValueTarget),
            Key(self.old_values_key, ValueEstimate),
        }

    @property
    def provides(self) -> set[Key]:
        return {Key(f"losses.{self.name}", LossScalar)}

    @property
    def constraints(self) -> list[str]:
        return [
            "clipped_surrogate_invariant(returns, values, old_values)"
        ]

    def validate(self, blackboard: Blackboard) -> None:
        values = blackboard.predictions.get("values_expected", blackboard.predictions["values"])
        returns = resolve_blackboard_path(blackboard, self.target_key)
        old_values = resolve_blackboard_path(blackboard, self.old_values_key)
        assert values.shape == returns.shape == old_values.shape, "Shape mismatch in ClippedValueLoss"

    def execute(self, blackboard: Blackboard) -> None:
        # 1. Extract inputs
        values = blackboard.predictions.get(
            "values_expected", blackboard.predictions["values"]
        )
        returns = resolve_blackboard_path(blackboard, self.target_key)
        old_values = resolve_blackboard_path(blackboard, self.old_values_key)

        # Ensure shapes match [B, T]
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        if returns.ndim == 3 and returns.shape[-1] == 1:
            returns = returns.squeeze(-1)
        if old_values.ndim == 3 and old_values.shape[-1] == 1:
            old_values = old_values.squeeze(-1)

        # 3. Compute losses
        v_loss_unclipped = (values - returns) ** 2
        v_clipped = old_values + torch.clamp(
            values - old_values, -self.clip_param, self.clip_param
        )
        v_loss_clipped = (v_clipped - returns) ** 2

        # PPO clipped value loss is the maximum of the two
        elementwise_loss = torch.max(v_loss_unclipped, v_loss_clipped)
        elementwise_loss = elementwise_loss * self.loss_factor

        # Pass through infrastructure
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        # Write out
        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()

        # Store elementwise loss for priority computation
        if "elementwise_losses" not in blackboard.meta:
            blackboard.meta["elementwise_losses"] = {}
        blackboard.meta["elementwise_losses"][self.name] = elementwise_loss
