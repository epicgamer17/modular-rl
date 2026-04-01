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
        modules: List[BaseLoss] = None,
        priority_computer: Optional[BasePriorityComputer] = None,
        # Explicit validator args
        minibatch_size: int = None,
        unroll_steps: int = 0,
        num_actions: int = 0,
        atom_size: int = 1,
        support_range: Optional[int] = None,
        representations: Optional[Dict[str, Any]] = None,
        shape_validator: Optional[ShapeValidator] = None,
    ):
        self.modules = modules or []
        self.priority_computer = priority_computer or NullPriorityComputer()
        self.representations = representations or {}
        if shape_validator is not None:
            self.shape_validator = shape_validator
        else:
            self.shape_validator = ShapeValidator(
                minibatch_size=minibatch_size,
                unroll_steps=unroll_steps,
                num_actions=num_actions,
                atom_size=atom_size,
                support_range=support_range,
            )

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
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
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

        # 2. Prediction & Target Formatting
        # This is the single point of truth for representation-based formatting.
        # It ensures losses stay blind to the underlying mathematical representations.
        formatted_target_keys = set()

        # A. Format Predictions (Expected Values & Distributions)
        for pred_key, rep in self.representations.items():
            if pred_key in predictions:
                if hasattr(rep, "to_expected_value"):
                    predictions[f"{pred_key}_expected"] = rep.to_expected_value(
                        predictions[pred_key]
                    )
                if hasattr(rep, "to_inference"):
                    predictions[f"{pred_key}_dist"] = rep.to_inference(
                        predictions[pred_key]
                    )

        # 3. Defaults and Secure Scaling
        # We no longer guess B and T. They are anchored by weights and gradient_scales.
        if weights is None:
            weights = targets["weights"]
        if gradient_scales is None:
            gradient_scales = targets["gradient_scales"]

        B = weights.shape[0]
        T = gradient_scales.shape[1]

        total_loss_dict = {
            m.optimizer_name: torch.tensor(0.0, device=device) for m in self.modules
        }
        loss_dict = {}
        all_elementwise_losses = {}

        # 4. Single-Pass Vectorized Execution
        for module in self.modules:
            # Blindly compute: decision logic now lives in the Factory/Registry
            # Compute ([B, T] elementwise loss, metrics_dict)
            elementwise_loss, module_metrics = module.compute_loss(predictions, targets)
            all_elementwise_losses[module.name] = elementwise_loss
            mask = module.get_mask(targets)

            # Aggregate metrics from the module
            loss_dict.update(module_metrics)

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
            all_elementwise_losses, predictions, targets
        )

        loss_dict["loss_pipeline_latency_ms"] = (
            time.perf_counter() - start_time
        ) * 1000
        return total_loss_dict, loss_dict, priorities
