import torch
import numpy as np
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from agents.learner.losses.base import BaseLoss
from agents.learner.losses.priorities import BasePriorityComputer, NullPriorityComputer
from agents.learner.losses.shape_validator import ShapeValidator

class LossPipeline:
    """
    Unified pipeline that handles both single-step (DQN) and sequence (MuZero) losses.
    Validated at initialization to ensure all required keys are present.
    """

    def __init__(
        self,
        config: Any,
        modules: List[BaseLoss],
        priority_computer: Optional[BasePriorityComputer] = None,
    ):
        self.config = config
        self.modules = modules
        self.priority_computer = priority_computer or NullPriorityComputer()
        self.shape_validator = ShapeValidator(config)

    def validate_dependencies(
        self, network_output_keys: set[str], target_keys: set[str]
    ) -> None:
        """
        Verify that the provided keys satisfy all module requirements.
        """
        for module in self.modules:
            missing_preds = module.required_predictions - network_output_keys
            missing_targets = module.required_targets - target_keys

            if missing_preds:
                raise ValueError(
                    f"Module {module.name} missing required predictions: {missing_preds}. "
                    f"Available: {network_output_keys}"
                )
            if missing_targets:
                raise ValueError(
                    f"Module {module.name} missing required targets: {missing_targets}. "
                    f"Available: {target_keys}"
                )

    def run(
        self,
        predictions: dict,
        targets: dict,
        context: dict = {},
        weights: Optional[torch.Tensor] = None,
        gradient_scales: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], torch.Tensor]:
        """
        Run the loss pipeline in a single vectorized pass across all sequence steps.
        """
        from modules.utils import scale_gradient

        # 1. Shape Validation and Latency Setup
        start_time = time.perf_counter()
        device = self.modules[0].device if self.modules else torch.device("cpu")
        self.shape_validator.validate(predictions, targets)

        # 2. Key/Format Normalization
        if not isinstance(predictions, dict):
            predictions = getattr(predictions, "_asdict", lambda: vars(predictions))()
        if not isinstance(targets, dict):
            targets = targets if isinstance(targets, dict) else vars(targets)

        # Determine dimensions B and T directly from the prediction tensors
        any_pred = next(p for p in predictions.values() if torch.is_tensor(p))
        B, T = any_pred.shape[:2]

        # 3. Defaults and Scaling
        if weights is None:
            weights = torch.ones(B, device=device)
        if gradient_scales is None:
            gradient_scales = torch.ones((1, T), device=device)

        context["full_targets"] = targets
        # Vectorized ChanceQ targets: shift values by 1
        if "values" in targets:
            v = targets["values"]
            v_next = torch.zeros_like(v)
            v_next[:, :-1] = v[:, 1:]
            context["target_values_next"] = v_next

        total_loss_dict = {
            m.optimizer_name: torch.tensor(0.0, device=device) for m in self.modules
        }
        loss_dict = {}
        all_elementwise_losses = {}

        # 4. Single-Pass Vectorized Execution
        for module in self.modules:
            # Blindly compute: decision logic now lives in the Factory/Registry
            # Compute [B, T] elementwise loss
            elementwise_loss = module.compute_loss(predictions, targets, context)
            all_elementwise_losses[module.name] = elementwise_loss
            mask = module.get_mask(targets)

            # Scale by Gradient Scales [1, T] and PER Weights [B, 1]
            scaled_loss = scale_gradient(elementwise_loss, gradient_scales)
            weighted_loss = scaled_loss * weights.reshape(B, 1)

            # Mask and Reduce (Sum-over-Mask)
            masked_weighted_loss = (weighted_loss * mask.float()).sum()
            valid_transition_count = mask.float().sum().clamp(min=1.0)

            # Final Transition-Averaged Loss [Scalar]
            total_scalar_loss = masked_weighted_loss / valid_transition_count

            total_loss_dict[module.optimizer_name] += total_scalar_loss
            loss_dict[module.name] = total_scalar_loss.item()

        # 5. Extract Priorities via standalone computer [B]
        priorities = self.priority_computer.compute(
            all_elementwise_losses, predictions, targets, context
        )

        # 6. Extract Auxiliary metrics from context (approx_kl, etc.)
        for key, value in context.items():
            if key in [
                "full_targets",
                "target_values_next",
                "has_valid_action_mask",
                "is_same_game",
            ]:
                continue
            if (
                isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], (int, float))
            ):
                loss_dict[key] = float(np.mean(value))

        loss_dict["loss_pipeline_latency_ms"] = (
            time.perf_counter() - start_time
        ) * 1000
        return total_loss_dict, loss_dict, priorities
