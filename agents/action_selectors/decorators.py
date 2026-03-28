import torch
import dataclasses
from agents.action_selectors.selectors import BaseActionSelector
from agents.action_selectors.types import InferenceResult
from torch.distributions import Categorical
from utils.schedule import create_schedule, Schedule, ScheduleConfig
from typing import Any, Dict, Optional, Tuple


class PPODecorator(BaseActionSelector):
    """
    Decorator that injects PPO-specific metadata (log_prob, value)
    into the selection result.
    """

    def __init__(self, inner_selector: BaseActionSelector):
        self.inner_selector = inner_selector

    def select_action(
        self,
        result: InferenceResult,
        info: Dict[str, Any],
        exploration: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # 1. Delegate selection to the inner selector
        action, metadata = self.inner_selector.select_action(
            result,
            info,
            exploration=exploration,
            **kwargs,
        )

        # 2. Inject PPO metadata
        dist = metadata.get("policy_dist")
        if dist is not None:
            metadata["log_prob"] = dist.log_prob(action).cpu()

        metadata["value"] = result.value.cpu()

        return action, metadata

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        """
        Pass parameter updates down to the inner selector.
        """
        self.inner_selector.update_parameters(params_dict)


class TemperatureSelector(BaseActionSelector):
    """
    Decorator that applies temperature scaling to action logits before selection.

    Supports both episode-step-based (local) and training-step-based (global) schedules,
    controlled by schedule_config.with_training_steps.

    Temperature = 0.0 collapses to argmax (greedy). Temperature = 1.0 is identity.
    Exploration=False forces temperature to 0.0 regardless of schedule.
    """

    def __init__(
        self, inner_selector: BaseActionSelector, schedule_config: ScheduleConfig
    ):
        super().__init__()
        self.inner_selector = inner_selector
        self.schedule_config = schedule_config
        self.schedule: Schedule = create_schedule(self.schedule_config)
        self.use_training_steps: bool = self.schedule_config.with_training_steps
        self._last_step: int = -1

    def _get_temperature(self, current_step: int, exploration: Optional[bool]) -> float:
        """Returns the temperature for a specific absolute step."""
        if exploration is False:
            return 0.0

        current_step = max(int(current_step), 0)
        if self.use_training_steps and current_step > self._last_step:
            self._last_step = current_step
        return float(self.schedule.get_value(step=current_step))

    def _resolve_temperature_steps(
        self,
        current_step: Any,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        default_step = max(self._last_step, 0) if self.use_training_steps else 0
        if current_step is None:
            current_step = default_step

        steps = torch.as_tensor(current_step, device=device, dtype=torch.long)
        if steps.dim() == 0:
            return steps.repeat(batch_size)

        steps = steps.reshape(-1)
        if steps.numel() == 1:
            return steps.repeat(batch_size)
        if steps.numel() != batch_size:
            raise ValueError(
                f"TemperatureSelector expected 1 or {batch_size} step values, got {steps.numel()}."
            )
        return steps

    def select_action(
        self,
        result: InferenceResult,
        info: Dict[str, Any],
        exploration: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Applies temperature scaling to result.logits / result.probs / result.q_values,
        writes the heated logits back to result.logits, then delegates to inner_selector.
        """
        if self.use_training_steps:
            current_step = kwargs.get("training_step", self._last_step)
        else:
            current_step = kwargs.get("episode_step", 0)

        # Resolve to logits (temperature only makes sense on logits)
        if result.logits is not None:
            logits = result.logits
        elif result.probs is not None:
            logits = torch.log(result.probs + 1e-8)
            logits = logits.masked_fill(result.probs == 0.0, -float("inf"))
        elif result.q_values is not None:
            logits = result.q_values  # Boltzmann exploration on Q-values
        else:
            raise ValueError(
                "TemperatureSelector requires result.logits, result.probs, or result.q_values"
            )

        mask = info.get("legal_moves_mask")
        needs_squeeze = logits.dim() == 1
        if needs_squeeze:
            logits = logits.unsqueeze(0)
            if mask is not None and torch.is_tensor(mask) and mask.dim() == 1:
                mask = mask.unsqueeze(0)

        steps = self._resolve_temperature_steps(
            current_step=current_step,
            batch_size=logits.shape[0],
            device=logits.device,
        )

        heated_rows = []
        for row_idx, row_logits in enumerate(logits):
            temp = self._get_temperature(int(steps[row_idx].item()), exploration)
            if temp == 0.0:
                select_logits = row_logits
                if mask is not None:
                    select_logits = torch.where(
                        mask[row_idx],
                        select_logits,
                        torch.tensor(
                            -1e9,
                            device=select_logits.device,
                            dtype=select_logits.dtype,
                        ),
                    )
                best_action = select_logits.argmax(dim=-1)
                heated_row = torch.full_like(row_logits, -float("inf"))
                heated_row[best_action] = 0.0
            elif temp != 1.0:
                heated_row = row_logits / temp
            else:
                heated_row = row_logits
            heated_rows.append(heated_row)

        logits = torch.stack(heated_rows, dim=0)
        if needs_squeeze:
            logits = logits.squeeze(0)

        # Return a new frozen result with the heated logits
        result = dataclasses.replace(result, logits=logits, probs=None)

        return self.inner_selector.select_action(
            result, info, exploration=exploration, **kwargs
        )

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        """
        Intercepts training_step broadcasts to track global step for the schedule.
        """
        if self.use_training_steps and "training_step" in params_dict:
            self._last_step = max(self._last_step, int(params_dict["training_step"]))
        self.inner_selector.update_parameters(params_dict)
