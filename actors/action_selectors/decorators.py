from typing import Any, Dict, Optional, Tuple
import torch
from actors.action_selectors.selectors import BaseActionSelector
from torch.distributions import Categorical
from utils.schedule import create_schedule, Schedule, ScheduleConfig


class PPODecorator(BaseActionSelector):
    """
    Decorator that injects PPO-specific metadata (log_prob, value)
    into the selection result.
    """

    def __init__(self, inner_selector: BaseActionSelector):
        self.inner_selector = inner_selector

    def select_action(
        self,
        predictions: Dict[str, torch.Tensor],
        info: Dict[str, Any],
        exploration: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # 1. Delegate selection to the inner selector
        action, metadata = self.inner_selector.select_action(
            predictions,
            info,
            exploration=exploration,
            **kwargs,
        )

        # 2. Inject PPO metadata
        dist = metadata.get("policy_dist")
        if dist is not None:
            metadata["log_prob"] = dist.log_prob(action).cpu()

        value = predictions.get("value")
        if value is not None:
            metadata["value"] = value.cpu()

        return action, metadata

    def mask_actions(
        self,
        values: torch.Tensor,
        legal_moves: Any,
        mask_value: float = -float("inf"),
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return self.inner_selector.mask_actions(values, legal_moves, mask_value, device)

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        """
        Pass parameter updates down to the inner selector.
        """
        self.inner_selector.update_parameters(params_dict)


class TemperatureSelector(BaseActionSelector):
    """
    Decorator that applies temperature scaling to action logits before selection.

    Supports both episode-step-based (local) and training-step-based (global) schedules.

    Temperature = 0.0 collapses to argmax (greedy). Temperature = 1.0 is identity.
    Exploration=False forces temperature to 0.0 regardless of schedule.
    """

    def __init__(
        self,
        inner_selector: BaseActionSelector,
        schedule: Schedule,
        use_training_steps: bool = False,
    ):
        super().__init__()
        self.inner_selector = inner_selector
        self.schedule = schedule
        self.use_training_steps = use_training_steps
        self._last_step: int = -1

    def _get_temperature(self, current_step: int, exploration: Optional[bool]) -> float:
        """Advances the schedule to current_step and returns the temperature."""
        if exploration is False:
            return 0.0

        if current_step > self._last_step:
            self.schedule.step(current_step - self._last_step)
            self._last_step = current_step
        elif current_step < self._last_step:
            # Reset on new episode or training step rollback
            self.schedule.reset()
            self.schedule.step(current_step)
            self._last_step = current_step

        return self.schedule.get_value()

    def select_action(
        self,
        predictions: Dict[str, torch.Tensor],
        info: Dict[str, Any],
        exploration: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Applies temperature scaling to logits / probs / q_values,
        writes the heated logits back to predictions['logits'], then delegates to inner_selector.
        """
        if self.use_training_steps:
            current_step = kwargs.get("training_step", self._last_step)
        else:
            current_step = kwargs.get("episode_step", 0)

        temp = self._get_temperature(current_step, exploration)

        pred_logits = predictions.get("logits")
        pred_probs = predictions.get("probs")
        pred_q_values = predictions.get("q_values")

        # Resolve to logits (temperature only makes sense on logits)
        if pred_logits is not None:
            logits = pred_logits
        elif pred_probs is not None:
            logits = torch.log(pred_probs + 1e-8)
            logits = logits.masked_fill(pred_probs == 0.0, -float("inf"))
        elif pred_q_values is not None:
            logits = pred_q_values  # Boltzmann exploration on Q-values
        else:
            raise ValueError(
                "TemperatureSelector requires logits, probs, or q_values in predictions"
            )

        # Apply temperature
        if temp == 0.0:
            mask = (
                info.get("legal_moves_mask", info.get("legal_moves")) if info else None
            )
            if mask is not None:
                logits = self.mask_actions(logits, mask)
            best_actions = logits.argmax(dim=-1)
            logits = torch.full_like(logits, -float("inf"))
            if logits.dim() == 1:
                logits[best_actions] = 0.0
            else:
                logits.scatter_(1, best_actions.unsqueeze(1), 0.0)
        elif temp != 1.0:
            logits = logits / temp

        # Write heated logits back; clear probs so downstream selector uses the new values
        predictions["logits"] = logits
        if "probs" in predictions:
            predictions["probs"] = None

        return self.inner_selector.select_action(
            predictions, info, exploration=exploration, **kwargs
        )

    def mask_actions(
        self,
        values: torch.Tensor,
        legal_moves: Any,
        mask_value: float = -float("inf"),
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return self.inner_selector.mask_actions(values, legal_moves, mask_value, device)

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        """
        Intercepts training_step broadcasts to track global step for the schedule.
        """
        if self.use_training_steps and "training_step" in params_dict:
            step = int(params_dict["training_step"])
            if step > self._last_step:
                self.schedule.step(step - self._last_step)
                self._last_step = step
        self.inner_selector.update_parameters(params_dict)
