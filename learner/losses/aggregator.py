import torch
import time
from typing import Any, Dict, List, Optional, Tuple

from learner.pipeline.base import PipelineComponent
from learner.core import Blackboard
from learner.losses.base import BaseLoss
from learner.losses.priorities import BasePriorityComputer, NullPriorityComputer
from learner.losses.shape_validator import ShapeValidator
from modules.utils import scale_gradient


class LossAggregator(PipelineComponent):
    """
    Unified pipeline that handles both single-step (DQN) and sequence (MuZero) losses.
    Reads individual losses from predictions/targets, sums them, and writes 'total_loss' 
    to the blackboard.
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

    def execute(self, blackboard: Blackboard) -> None:
        """
        Execute the loss pipeline in a single vectorized pass reading from Blackboard.
        Memory layout is structured to maintain GPU contiguous checks for throughput.
        """
        start_time = time.perf_counter()
        device = self.modules[0].device if self.modules else torch.device("cpu")
        
        # 1. Shape Validation
        self.shape_validator.validate(blackboard.predictions, blackboard.targets)

        # 2. Prediction & Target Formatting
        for pred_key, rep in self.representations.items():
            if pred_key in blackboard.predictions:
                if hasattr(rep, "to_expected_value"):
                    blackboard.predictions[f"{pred_key}_expected"] = rep.to_expected_value(
                        blackboard.predictions[pred_key]
                    )
                if hasattr(rep, "to_inference"):
                    blackboard.predictions[f"{pred_key}_dist"] = rep.to_inference(
                        blackboard.predictions[pred_key]
                    )

        # 3. Secure Weights & Gradient Scales from Meta (Truth Source)
        weights = blackboard.meta.get("weights")
        gradient_scales = blackboard.meta.get("gradient_scales")
        
        if weights is None or gradient_scales is None:
            raise ValueError(
                f"LossAggregator requires 'weights' and 'gradient_scales' in blackboard.meta. "
                f"Did you run UniversalInfrastructureComponent? Meta keys: {list(blackboard.meta.keys())}"
            )

        B = weights.shape[0]
        T = gradient_scales.shape[1]

        total_loss_dict = {
            m.optimizer_name: torch.tensor(0.0, device=device) for m in self.modules
        }
        loss_dict = {}
        all_elementwise_losses = {}

        total_module_loss = 0.0
        # 4. Single-Pass Vectorized Execution
        for module in self.modules:
            # Blindly compute: decision logic now lives in the Factory/Registry
            elementwise_loss, module_metrics = module.compute_loss(
                blackboard.predictions, blackboard.targets
            )
            all_elementwise_losses[module.name] = elementwise_loss
            mask = module.get_mask(blackboard.targets)

            loss_dict.update(module_metrics)

            # Scale and Weight
            scaled_loss = scale_gradient(elementwise_loss, gradient_scales)
            weighted_loss = scaled_loss * weights.reshape(B, 1)

            # Mask and Reduce
            masked_weighted_loss = (weighted_loss * mask.float()).sum()
            valid_transition_count = mask.float().sum().clamp(min=1.0)
            
            total_scalar_loss = masked_weighted_loss / valid_transition_count

            total_loss_dict[module.optimizer_name] += total_scalar_loss
            scalar_loss_val = total_scalar_loss.item()
            loss_dict[module.name] = scalar_loss_val
            total_module_loss += scalar_loss_val

        # 5. Extract Priorities
        priorities = self.priority_computer.compute(
            all_elementwise_losses, blackboard.predictions, blackboard.targets
        )

        loss_dict["loss_pipeline_latency_ms"] = (
            time.perf_counter() - start_time
        ) * 1000
        loss_dict["loss"] = total_module_loss

        # Optional contiguous memory enforce before passing gradient nodes via blackboard
        blackboard.losses = {k: v.contiguous() for k, v in total_loss_dict.items()}
        
        blackboard.meta.update(loss_dict)
        blackboard.meta["priorities"] = priorities
