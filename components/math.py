import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, TYPE_CHECKING
from core import PipelineComponent
from core import Blackboard
from modules.utils import scale_gradient

if TYPE_CHECKING:
    from modules.agent_nets.base import BaseAgentNetwork


# TODO: get rid of this hacky stuff
def apply_infrastructure(
    elementwise_loss: torch.Tensor,
    blackboard: Blackboard,
    mask_key: str,
) -> torch.Tensor:
    """
    Applies standard infrastructure to a [B, T] elementwise loss:
    1. Gradient Scaling (depth-based)
    2. Weights (Importance Sampling)
    3. Masking
    4. Reduction to Mean Scalar
    """
    # 1. Secure Weights & Gradient Scales from Meta (Truth Source)
    weights = blackboard.meta.get("weights")
    gradient_scales = blackboard.meta.get("gradient_scales")
    masks = blackboard.targets.get(mask_key)

    if weights is None or gradient_scales is None or masks is None:
        # Fallback for simple environments or missing infra
        B, T = elementwise_loss.shape[:2]
        device = elementwise_loss.device
        weights = weights if weights is not None else torch.ones(B, device=device)
        gradient_scales = (
            gradient_scales
            if gradient_scales is not None
            else torch.ones((1, T), device=device)
        )
        masks = (
            masks
            if masks is not None
            else torch.ones((B, T), device=device, dtype=torch.bool)
        )

    B = weights.shape[0]
    T = gradient_scales.shape[1]

    # Normalize elements
    elementwise_loss = elementwise_loss.reshape(B, T)

    # 2. Scale and Weight
    scaled_loss = scale_gradient(elementwise_loss, gradient_scales)
    weighted_loss = scaled_loss * weights.reshape(B, 1)

    # 3. Mask and Reduce
    masked_weighted_loss = (weighted_loss * masks.float()).sum()
    valid_transition_count = masks.float().sum().clamp(min=1.0)

    return masked_weighted_loss / valid_transition_count


# TODO: make this a folder instead of a file? or is it good as a file?
class LossAggregatorComponent(PipelineComponent):
    """
    Reads individual loss keys, applies predefined weights, sums them,
    and writes total_loss to the Blackboard.
    """

    def __init__(self, loss_weights: Dict[str, float], optimizer_key: str = "default"):
        self.loss_weights = loss_weights
        self.optimizer_key = optimizer_key

    def execute(self, blackboard: Blackboard) -> None:
        if not blackboard.losses:
            return

        total_loss = None

        for loss_name, weight in self.loss_weights.items():
            if loss_name in blackboard.losses:
                weighted_loss = weight * blackboard.losses[loss_name]
                if total_loss is None:
                    total_loss = weighted_loss
                else:
                    total_loss = total_loss + weighted_loss

        if total_loss is None:
            return

        # Write the final backward-ready tensor to the blackboard
        # Grouped by optimizer_key in case of disjoint networks (e.g., separate Actor/Critic opts)
        if "total_loss" not in blackboard.losses:
            blackboard.losses["total_loss"] = {}

        blackboard.losses["total_loss"][self.optimizer_key] = total_loss


class OptimizerStepComponent(PipelineComponent):
    """
    Sits at the end of the pipeline. Calls backward(), applies clipping, and steps.
    """

    def __init__(
        self,
        agent_network: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        max_grad_norm: Optional[float] = None,
    ):
        self.agent_network = agent_network
        self.optimizers = optimizers
        self.max_grad_norm = max_grad_norm

    def execute(self, blackboard: Blackboard) -> None:
        total_losses = blackboard.losses.get("total_loss", {})

        for opt_key, loss_tensor in total_losses.items():
            opt = self.optimizers[opt_key]

            opt.zero_grad(set_to_none=True)
            loss_tensor.backward()

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.agent_network.parameters(), self.max_grad_norm
                )

            opt.step()


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

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions["values"]
        targets = blackboard.targets[self.target_key]

        B, T = preds.shape[:2]

        # Flatten B, T for loss function
        raw_loss = self.loss_fn(
            preds.flatten(0, 1), targets.flatten(0, 1), reduction="none"
        )

        # Reshape to [B, T]
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)
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

    def execute(self, blackboard: Blackboard) -> None:
        # 1. Extract inputs
        values = blackboard.predictions.get(
            "values_expected", blackboard.predictions["values"]
        )
        returns = blackboard.targets[self.target_key]
        old_values = blackboard.targets[self.old_values_key]

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


class PolicyLoss(PipelineComponent):
    """Policy prediction loss module."""

    def __init__(
        self,
        loss_fn: Any,
        loss_factor: float = 1.0,
        mask_key: str = "policy_mask",
        name: str = "policy_loss",
    ):
        self.loss_fn = loss_fn
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions["policies"]
        targets = blackboard.targets["policies"]

        B, T = preds.shape[:2]

        # Flatten B, T
        raw_loss = self.loss_fn(
            preds.flatten(0, 1), targets.flatten(0, 1), reduction="none"
        )
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)
        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor

        # Pass through infrastructure
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        # Logging
        with torch.no_grad():
            log_q = F.log_softmax(preds, dim=-1)
            log_p = torch.log(targets + 1e-10)
            kl = (targets * (log_p - log_q)).sum(dim=-1).mean()
            blackboard.meta["approx_kl"] = kl.item()

        # Write out
        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class ClippedSurrogateLoss(PipelineComponent):
    """PPO Clipped Surrogate Policy Loss."""

    def __init__(
        self,
        clip_param: float,
        entropy_coefficient: float,
        mask_key: str = "policy_mask",
        name: str = "policy_loss",
    ):
        self.clip_param = clip_param
        self.entropy_coefficient = entropy_coefficient
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        policy_logits = blackboard.predictions["policies"]
        actions = blackboard.targets["actions"]
        old_log_probs = blackboard.targets["old_log_probs"]
        advantages = blackboard.targets["advantages"]

        dist = torch.distributions.Categorical(logits=policy_logits)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )

        entropy = dist.entropy()
        elementwise_loss = -torch.min(surr1, surr2) - self.entropy_coefficient * entropy

        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean()
            blackboard.meta["approx_kl"] = approx_kl.item()

        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class ImitationLoss(PipelineComponent):
    """Imitation learning loss (Behavioral Cloning)."""

    def __init__(
        self,
        loss_fn: Any,
        loss_factor: float = 1.0,
        mask_key: str = "policy_mask",
        target_key: str = "policies",
        name: str = "policy_loss",
    ):
        self.loss_fn = loss_fn
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.target_key = target_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        pred = blackboard.predictions["policies"]
        target = blackboard.targets[self.target_key]

        B, T = pred.shape[:2]

        if target.shape == pred.shape:
            flat_target = target.flatten(0, 1)
        else:
            flat_target = target.flatten(0, 1).long()

        raw_loss = self.loss_fn(pred.flatten(0, 1), flat_target, reduction="none")
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)
        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor

        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class RewardLoss(PipelineComponent):
    """Reward prediction loss module."""

    def __init__(
        self,
        loss_fn: Any,
        loss_factor: float = 1.0,
        mask_key: str = "reward_mask",
        name: str = "reward_loss",
    ):
        self.loss_fn = loss_fn
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions["rewards"]
        targets = blackboard.targets["rewards"]

        B, T = preds.shape[:2]

        raw_loss = self.loss_fn(
            preds.flatten(0, 1), targets.flatten(0, 1), reduction="none"
        )
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)
        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor

        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class ToPlayLoss(PipelineComponent):
    """To-play prediction loss module."""

    def __init__(
        self,
        loss_fn: Any = F.binary_cross_entropy_with_logits,
        loss_factor: float = 1.0,
        mask_key: str = "to_play_mask",
        name: str = "to_play_loss",
    ):
        self.loss_fn = loss_fn
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions.get("to_plays")
        targets = blackboard.targets.get("to_plays")

        if preds is None or targets is None:
            return

        B, T = preds.shape[:2]
        raw_loss = self.loss_fn(
            preds.flatten(0, 1), targets.flatten(0, 1), reduction="none"
        )
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)
        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor

        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class QBootstrappingLoss(PipelineComponent):
    """Standard TD target loss for Q-learning."""

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

        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class ConsistencyLoss(PipelineComponent):
    """Latent consistency loss (EfficientZero style)."""

    def __init__(
        self,
        loss_factor: float = 1.0,
        mask_key: str = "masks",
        name: str = "consistency_loss",
    ):
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions.get("projected_latents")
        targets = blackboard.targets.get("consistency_targets")

        if preds is None or targets is None:
            return

        B, T = preds.shape[:2]

        preds_norm = F.normalize(preds, p=2.0, dim=-1)
        targets_norm = F.normalize(targets, p=2.0, dim=-1)

        elementwise_loss = -(preds_norm * targets_norm).sum(dim=-1) * self.loss_factor
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class SigmaLoss(PipelineComponent):
    """Loss for sigma prediction (stochastic MuZero)."""

    def __init__(
        self,
        loss_factor: float = 1.0,
        mask_key: str = "masks",
        name: str = "sigma_loss",
    ):
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions.get("sigma_logits")
        targets = blackboard.targets.get("sigmas")

        if preds is None or targets is None:
            return

        B, T = preds.shape[:2]
        raw_loss = F.cross_entropy(
            preds.flatten(0, 1), targets.flatten(0, 1), reduction="none"
        )
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)

        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class CommitmentLoss(PipelineComponent):
    """VQ-VAE commitment cost for encoder (stochastic MuZero)."""

    def __init__(
        self,
        loss_factor: float = 1.0,
        mask_key: str = "masks",
        name: str = "commitment_loss",
    ):
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        elementwise_loss = blackboard.predictions.get("commitment_loss")
        if elementwise_loss is None:
            return

        elementwise_loss = elementwise_loss * self.loss_factor
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class LatentConsistencyComponent(PipelineComponent):
    """EfficientZero consistency loss target builder."""

    def __init__(self, agent_network: nn.Module):
        self.agent_network = agent_network

    def execute(self, blackboard: Blackboard) -> None:
        real_obs = blackboard.data["unroll_observations"].float()
        batch_size, unroll_len = real_obs.shape[:2]
        flat_obs = real_obs.flatten(0, 1)

        with torch.no_grad():
            initial_out = self.agent_network.obs_inference(flat_obs)
            real_latents = initial_out.network_state.dynamics
            proj_targets = self.agent_network.project(real_latents, grad=False)
            normalized_targets = torch.nn.functional.normalize(
                proj_targets, p=2.0, dim=-1, eps=1e-5
            )

        blackboard.targets["consistency_targets"] = normalized_targets.reshape(
            batch_size, unroll_len, -1
        ).detach()


# TODO: clean this up
class ShapeValidator:
    """Validates tensor shapes."""

    def __init__(
        self,
        minibatch_size: int,
        unroll_steps: int = 0,
        num_actions: int = 0,
        atom_size: int = 1,
        support_range: Optional[int] = None,
    ):
        self.B = minibatch_size
        self.K = unroll_steps
        self.num_actions = num_actions
        self.atom_size = atom_size
        if self.atom_size == 1 and support_range is not None:
            self.atom_size = (support_range * 2) + 1
        self.T = self.K + 1

    def validate(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> None:
        self.validate_predictions(predictions)
        for key, tensor in targets.items():
            if torch.is_tensor(tensor):
                self._check_shape_strict(key, tensor, is_prediction=False)

    def validate_predictions(self, predictions: Dict[str, torch.Tensor]) -> None:
        for key, tensor in predictions.items():
            if torch.is_tensor(tensor):
                self._check_shape_strict(key, tensor, is_prediction=True)

    def _check_shape_strict(
        self, key: str, tensor: torch.Tensor, is_prediction: bool
    ) -> None:
        if key in ["weights", "gradient_scales", "metrics"]:
            return
        shape = list(tensor.shape)
        prefix = f"[{'Prediction' if is_prediction else 'Target'}] '{key}'"
        assert shape[0] == self.B, f"{prefix} B mismatch: {self.B} vs {shape[0]}"
        assert len(shape) >= 2, f"{prefix} dim < 2: {shape}"
        assert shape[1] == self.T, f"{prefix} T mismatch: {self.T} vs {shape[1]}"


class ShapeValidatorComponent(PipelineComponent):
    """Pipeline Component wrapper for ShapeValidator."""

    def __init__(self, validator: ShapeValidator):
        self.validator = validator

    def execute(self, blackboard: Blackboard) -> None:
        self.validator.validate(blackboard.predictions, blackboard.targets)


class MetricEarlyStopComponent(PipelineComponent):
    """Stops training if a metric (e.g. reward) exceeds a threshold."""
    def __init__(self, threshold: float, metric_key: str = "approx_kl"):
        self.metric_key = metric_key
        self.threshold = threshold

    def execute(self, blackboard: Blackboard) -> None:
        # Check both meta and losses for the metric
        val = blackboard.meta.get(self.metric_key)
        if val is None:
            val = blackboard.losses.get(self.metric_key)

        if val is not None:
            # Handle both tensors and scalars
            scalar_val = val.item() if torch.is_tensor(val) else val
            if scalar_val >= self.threshold:
                print(f"Early Stopping: {self.metric_key} reached {scalar_val} >= {self.threshold}")
                blackboard.meta["stop_execution"] = True


class MPSCacheClearComponent(PipelineComponent):
    """Clears MPS cache to avoid memory leaks on macOS."""
    def __init__(self, device: torch.device, interval: int = 100):
        self.device = device
        self.interval = interval
        self.step_count = 0

    def execute(self, blackboard: Blackboard) -> None:
        if self.device.type == "mps":
            self.step_count += 1
            if self.step_count % self.interval == 0:
                torch.mps.empty_cache()


class DeviceTransferComponent(PipelineComponent):
    """Transfers the entire data dictionary to the specified device."""

    def __init__(self, device: torch.device):
        self.device = device

    def execute(self, blackboard: Blackboard) -> None:
        for k, v in blackboard.data.items():
            if torch.is_tensor(v):
                blackboard.data[k] = v.to(self.device)
