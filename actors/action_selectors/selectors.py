import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
from utils.schedule import Schedule

# Constant for default epsilon
DEFAULT_EPSILON = 0.05


class BaseActionSelector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select_action(
        self,
        predictions: Dict[str, torch.Tensor],
        info: Dict[str, Any],
        exploration: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError  # pragma: no cover

    def mask_actions(
        self,
        values: torch.Tensor,
        legal_moves: Any,
        mask_value: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Masks illegal actions in the given values (logits, probs, or Q-values).

        Args:
            values: The tensor to mask [B, A] or [A].
            legal_moves: List of legal move indices or list of lists for batched input.
            mask_value: The value to use for masking (defaults to self.config.default_mask_value).
            device: Optional device.

        Returns:
            The masked tensor.
        """
        if legal_moves is None:
            return values
        if mask_value is None:
            mask_value = -float("inf")

        if device is None:
            device = values.device

        # Core masking logic (adapted from utils.action_mask)
        mask = torch.zeros_like(values, dtype=torch.bool).to(device)

        if values.dim() == 1:
            if isinstance(legal_moves, (list, np.ndarray, torch.Tensor)):
                mask[legal_moves] = True
            else:
                raise ValueError(
                    f"For 1D actions, legal_moves must be an iterable of indices, got {type(legal_moves)}"
                )
        elif values.dim() == 2:
            # Batch of legal moves: [[...], [...]]
            # Special case: if batch size is 1 and legal_moves is a single list of moves OR a 1D tensor mask
            if values.shape[0] == 1 and len(legal_moves) > 0:
                # If it's a list, check if the first element is a list to determine nesting
                is_nested = isinstance(legal_moves[0], (list, np.ndarray)) or (
                    torch.is_tensor(legal_moves) and legal_moves.dim() > 1
                )

                if not is_nested:
                    mask[0, legal_moves] = True
                else:
                    mask[0, legal_moves[0]] = True
            else:
                for i, legal in enumerate(legal_moves):
                    if legal is not None:
                        mask[i, legal] = True
        else:
            raise ValueError(
                f"mask_actions expects 1D or 2D tensor, got {values.dim()}D"
            )

        return torch.where(mask, values, torch.tensor(mask_value, device=device))

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        """
        Updates the internal parameters of the selector.

        Args:
            params_dict: Dictionary containing parameter updates.
        """
        pass  # pragma: no cover


class ActionSelector(BaseActionSelector):
    """
    Unified action selector for discrete action spaces.

    Combines categorical sampling, temperature scaling, and greedy (argmax) selection.

    Logic:
    - If temperature > 0: Samples from a Categorical distribution scaled by temperature.
    - If temperature == 0: Selects the action with the highest logit/value (Argmax).
    """

    LOG_EPSILON = 1e-10

    def __init__(
        self,
        input_key: str,
        temperature: float = 1.0,
        schedule: Optional[Schedule] = None,
        use_training_steps: bool = False,
    ):
        super().__init__()
        self.input_key = input_key
        self.temperature = temperature
        self.schedule = schedule
        self.use_training_steps = use_training_steps
        self._last_step = -1

    def _get_temperature(self, current_step: int, exploration: Optional[bool]) -> float:
        """Advances the schedule and returns the current temperature."""
        if exploration is False:
            return 0.0

        if self.schedule is None:
            return self.temperature

        if current_step != self._last_step:
            if current_step > self._last_step:
                self.schedule.step(
                    max(1, current_step - self._last_step)
                    if self._last_step >= 0
                    else current_step
                )
            else:
                self.schedule.reset()
                self.schedule.step(current_step)
            self._last_step = current_step

        return self.schedule.get_value()

    def select_action(
        self,
        predictions: Union[Dict[str, torch.Tensor], "InferenceOutput"],
        info: Dict[str, Any],
        exploration: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Samples an action based on temperature.
        """
        if self.use_training_steps:
            current_step = kwargs.get("training_step", self._last_step)
        else:
            current_step = kwargs.get("episode_step", 0)

        temp = self._get_temperature(current_step, exploration)

        # Resolve values from predictions based on input_key
        # predictions can be a dictionary (from PolicySource) or an InferenceOutput object
        values = None
        if isinstance(predictions, dict):
            values = predictions.get(self.input_key)
            if values is None and "extra_metadata" in predictions:
                values = predictions["extra_metadata"].get(self.input_key)
        else:
            if hasattr(predictions, self.input_key):
                values = getattr(predictions, self.input_key)
            elif (
                hasattr(predictions, "extras")
                and predictions.extras
                and self.input_key in predictions.extras
            ):
                values = predictions.extras[self.input_key]
            elif (
                hasattr(predictions, "metadata")
                and predictions.metadata
                and self.input_key in predictions.metadata
            ):
                # Legacy support for 'metadata' field if it exists in some custom objects
                values = predictions.metadata[self.input_key]

        if values is None:
            available = (
                list(predictions.keys())
                if isinstance(predictions, dict)
                else dir(predictions)
            )
            raise KeyError(
                f"Key '{self.input_key}' not found in predictions. Available: {available}"
            )

        if values is None:
            raise ValueError(f"Values for key '{self.input_key}' are None")

        is_prob = self.input_key == "probs"

        mask = info.get("legal_moves_mask", info.get("legal_moves"))

        # Apply masking
        if is_prob:
            masked_values = self.mask_actions(values, mask, mask_value=0.0)
        else:
            masked_values = self.mask_actions(values, mask, mask_value=-float("inf"))

        if temp <= 0.0:
            # Greedy / Argmax selection
            action = torch.argmax(masked_values, dim=-1)
            dist = None
        else:
            # Categorical sampling
            from torch.distributions import Categorical

            if is_prob:
                temp_logits = torch.log(masked_values + self.LOG_EPSILON)
                dist = Categorical(logits=temp_logits / temp)
            else:
                dist = Categorical(logits=masked_values / temp)
            action = dist.sample()

        # Prepare metadata
        metadata = {"temperature": temp}
        if dist is not None:
            metadata["policy_dist"] = dist
            metadata["policy"] = dist.probs.detach()
        elif is_prob:
            metadata["policy"] = masked_values.detach()
        else:
            metadata["policy"] = (
                torch.nn.functional.one_hot(action, num_classes=values.shape[-1])
                .float()
                .detach()
            )

        return action, metadata
