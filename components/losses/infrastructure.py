from typing import Any, Dict, Optional, Set, Dict as TypedDict
import torch
import torch.nn as nn
from core import PipelineComponent, Blackboard
from core.contracts import Key, LossScalar, WriteMode, SemanticType, Epsilon, Metric, Observation
from core.validation import assert_in_blackboard
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
        self._provides = {Key("meta.epsilon", Epsilon): WriteMode.NEW}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, WriteMode]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """No inputs to validate; this component is a pure generator."""
        pass

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        if self.decay_steps > 0:
            epsilon = self.initial_epsilon - (
                (self.initial_epsilon - self.min_epsilon)
                * min(1.0, self.current_step / self.decay_steps)
            )
        else:
            epsilon = self.min_epsilon

        self.current_step += 1
        return {"meta.epsilon": epsilon}


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
    # [1] Shape Discovery (Primary source of truth is elementwise_loss)
    assert elementwise_loss.dim() >= 2, f"elementwise_loss must be at least 2D [B, T, ...], got {elementwise_loss.shape}"
    B, T = elementwise_loss.shape[:2]
    device = elementwise_loss.device

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

    # Graceful Defaults
    if weights is None:
        weights = torch.ones(B, device=device)
    if gradient_scales is None:
        # Default to no gradient scaling
        gradient_scales = torch.ones((1, T), device=device)
    if masks is None:
        # Default to full tensor if no mask is found
        masks = torch.ones((B, T), device=device, dtype=torch.bool)

    # [2] Strict Shape Enforcement
    # Weights must be [B] or [B, 1] (we'll reshape to [B, 1] for broadcasting to T)
    assert weights.shape[0] == B, f"Weights batch size {weights.shape[0]} does not match loss batch size {B}"
    
    # Masks must be EXACTLY [B, T] to prevent cross-batch/cross-time mixing
    assert masks.shape == (B, T), (
        f"Mask shape {masks.shape} does not match elementwise_loss shape ({(B, T)}). "
        f"Strict matching required to prevent silent broadcasting bugs."
    )

    # Gradient scales must be [1, T] or [B, T]
    assert gradient_scales.shape == (1, T) or gradient_scales.shape == (B, T), (
        f"Gradient scales shape {gradient_scales.shape} must be either (1, {T}) or ({B}, {T})"
    )

    # Normalize elementwise_loss to [B, T]
    # This enforces that there are exactly B * T elements (no hidden indices)
    assert elementwise_loss.numel() == B * T, (
        f"elementwise_loss has {elementwise_loss.numel()} elements, but expected B={B} * T={T} = {B*T}. "
        f"Shape: {elementwise_loss.shape}. Elementwise loss must be exactly [B, T] after broadcasting."
    )
    elementwise_loss = elementwise_loss.reshape(B, T)

    # 2. Scale and Weight
    scaled_loss = scale_gradient(elementwise_loss, gradient_scales)
    weighted_loss = scaled_loss * weights.reshape(B, 1)

    # 3. Mask and Reduce
    # Double check shape before multiplication
    assert weighted_loss.shape == masks.shape, f"Weighted loss {weighted_loss.shape} and masks {masks.shape} must match exactly"
    
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
            Key("losses.total_loss", LossScalar): WriteMode.NEW,
        }

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, WriteMode]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures at least some of the declared loss keys exist."""
        available = set(blackboard.losses.keys()) if blackboard.losses else set()
        declared = set(self.loss_weights.keys())
        assert available & declared, (
            f"LossAggregatorComponent: none of the declared losses {declared} "
            f"found in blackboard.losses (available: {available})"
        )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        if not blackboard.losses:
            return {}

        total_loss = None

        for loss_name, weight in self.loss_weights.items():
            if loss_name in blackboard.losses:
                weighted_loss = weight * blackboard.losses[loss_name]
                if total_loss is None:
                    total_loss = weighted_loss
                else:
                    total_loss = total_loss + weighted_loss

        if total_loss is None:
            return {}

        # Return as a dictionary under a single key to avoid write_blackboard_path overwriting logic
        return {
            "losses.total_loss": {self.optimizer_key: total_loss}
        }


class OptimizerStepComponent(PipelineComponent):
    """
    Sits at the end of the pipeline. Calls backward(), applies clipping, and steps.
    """
    required = True



    def __init__(
        self,
        agent_network: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        max_grad_norm: Optional[float] = None,
    ):
        self.agent_network = agent_network
        self.optimizers = optimizers
        self.max_grad_norm = max_grad_norm
        self._requires = {Key("losses.total_loss", LossScalar)}
        self._provides = {Key("meta.optimizer_steps", Metric): "optional"}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, WriteMode]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        assert_in_blackboard(blackboard, "losses.total_loss")

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
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
        
        return {"meta.optimizer_steps": 1}





class MetricEarlyStopComponent(PipelineComponent):
    """Stops training if a metric (e.g. reward) exceeds a threshold."""
    def __init__(self, threshold: float, metric_key: str = "approx_kl"):
        self.metric_key = metric_key
        self.threshold = threshold
        # Only require meta key - the execute checks both meta and losses as fallback
        self._requires = {Key(f"meta.{self.metric_key}", Metric)}
        self._provides = {Key("meta.stop_execution", Metric): "optional"}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, WriteMode]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures at least one source for the metric exists."""
        val = blackboard.meta.get(self.metric_key)
        if val is None:
            val = blackboard.losses.get(self.metric_key)
        assert val is not None, (
            f"MetricEarlyStopComponent: metric '{self.metric_key}' not found "
            f"in blackboard.meta or blackboard.losses"
        )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        # Check both meta and losses for the metric (one is guaranteed by validate)
        val = blackboard.meta.get(self.metric_key, blackboard.losses.get(self.metric_key))

        if val is not None:
            # Handle both scalars and tensors
            scalar_val = val.item() if torch.is_tensor(val) else val
            if scalar_val >= self.threshold:
                print(f"Early Stopping: {self.metric_key} reached {scalar_val} >= {self.threshold}")
                return {"meta.stop_execution": True}
        
        return {}


class MPSCacheClearComponent(PipelineComponent):
    """Clears MPS cache to avoid memory leaks on macOS."""
    def __init__(self, device: torch.device, interval: int = 100):
        self.device = device
        self.interval = interval
        self.step_count = 0
        self._requires = set()
        self._provides = {}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, WriteMode]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """No inputs to validate; this component is a side-effect-only utility."""
        pass

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        if self.device.type == "mps":
            self.step_count += 1
            if self.step_count % self.interval == 0:
                torch.mps.empty_cache()
        return {}


class DeviceTransferComponent(PipelineComponent):
    """Transfers the entire data dictionary to the specified device."""

    def __init__(self, device: torch.device):
        self.device = device
        self._requires = {Key("data", Observation)}
        self._provides = {Key("data", Observation): "overwrite"}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, WriteMode]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures data dict exists and is non-empty."""
        assert blackboard.data, "DeviceTransferComponent: blackboard.data is empty"

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        updates = {}
        for k, v in blackboard.data.items():
            if torch.is_tensor(v):
                updates[f"data.{k}"] = v.to(self.device)
        return updates
