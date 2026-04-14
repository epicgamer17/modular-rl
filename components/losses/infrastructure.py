from typing import Any, Dict, Optional, Set
import torch
import torch.nn as nn
from core import PipelineComponent, Blackboard
from core.contracts import Key, LossScalar, SemanticType
from modules.utils import scale_gradient


class EpsilonDecayComponent(PipelineComponent):
    """
    Component for linear epsilon decay.
    Writes current epsilon to blackboard.meta["epsilon"].
    """

    def __init__(
        self,
        initial_epsilon: float,
        min_epsilon: float,
        decay_steps: int,
    ):
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_steps = decay_steps
        self.current_step = 0
        self._requires = set()
        self._provides = {Key("meta.epsilon", SemanticType)}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
        if self.decay_steps > 0:
            epsilon = self.initial_epsilon - (
                (self.initial_epsilon - self.min_epsilon)
                * min(1.0, self.current_step / self.decay_steps)
            )
        else:
            epsilon = self.min_epsilon

        blackboard.meta["epsilon"] = epsilon
        self.current_step += 1


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
    # 1. Weights from data (yielded by Sampler) or meta (Truth Source)
    weights = blackboard.data.get("weights")
    if weights is None:
        weights = blackboard.meta.get("weights")

    # 2. Gradient scales from meta
    gradient_scales = blackboard.meta.get("gradient_scales")

    # 3. Masks from targets (e.g. PivotComponent) or data (Sampler)
    masks = blackboard.targets.get(mask_key)
    if masks is None:
        masks = blackboard.data.get(mask_key)

    B, T = elementwise_loss.shape[:2]
    device = elementwise_loss.device

    # Graceful Defaults (No longer using UniversalInfrastructureComponent)
    if weights is None:
        weights = torch.ones(B, device=device)
    if gradient_scales is None:
        # Default to no gradient scaling
        gradient_scales = torch.ones((1, T), device=device)
    if masks is None:
        # Default to full tensor if no mask is found
        masks = torch.ones((B, T), device=device, dtype=torch.bool)

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


class LossAggregatorComponent(PipelineComponent):
    """
    Reads individual loss keys, applies predefined weights, sums them,
    and writes total_loss to the Blackboard.
    """

    def __init__(self, loss_weights: Dict[str, float], optimizer_key: str = "default"):
        self.loss_weights = loss_weights
        self.optimizer_key = optimizer_key
        
        # Deterministic contracts computed at initialization
        self._requires = {Key(f"losses.{name}", LossScalar) for name in self.loss_weights.keys()}
        self._provides = {
            Key(f"losses.total_loss.{self.optimizer_key}", LossScalar),
            Key("losses.total_loss", SemanticType)
        }

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        pass

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
        self._requires = {Key("losses.total_loss", SemanticType)}
        self._provides = set()

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        assert "total_loss" in blackboard.losses

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
        self._requires = {Key("predictions", SemanticType), Key("targets", SemanticType)}
        self._provides = set()

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
        self.validator.validate(blackboard.predictions, blackboard.targets)


class MetricEarlyStopComponent(PipelineComponent):
    """Stops training if a metric (e.g. reward) exceeds a threshold."""
    def __init__(self, threshold: float, metric_key: str = "approx_kl"):
        self.metric_key = metric_key
        self.threshold = threshold
        self._requires = {
            Key(f"meta.{self.metric_key}", SemanticType),
            Key(f"losses.{self.metric_key}", SemanticType)
        }
        self._provides = {Key("meta.stop_execution", SemanticType)}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        pass

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
        self._requires = set()
        self._provides = set()

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
        if self.device.type == "mps":
            self.step_count += 1
            if self.step_count % self.interval == 0:
                torch.mps.empty_cache()


class DeviceTransferComponent(PipelineComponent):
    """Transfers the entire data dictionary to the specified device."""

    def __init__(self, device: torch.device):
        self.device = device
        self._requires = {Key("data", SemanticType)}
        self._provides = {Key("data", SemanticType)}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
        for k, v in blackboard.data.items():
            if torch.is_tensor(v):
                blackboard.data[k] = v.to(self.device)
