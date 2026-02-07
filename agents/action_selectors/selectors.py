from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import numpy as np

# Constant for default epsilon
DEFAULT_EPSILON = 0.05


class ActionSelector(ABC):
    """
    Abstract interface for action selection logic.
    All action selectors should inherit from this class.
    """

    @abstractmethod
    def select(self, values: Any, **kwargs) -> Any:
        """
        Selects an action based on model values/logits.

        Args:
            values: Model output (logits, probabilities, or q-values).
            **kwargs: Additional parameters for selection (e.g., temperature, exploration).

        Returns:
            The selected action(s).
        """
        pass

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        """
        Updates the internal parameters of the selector.

        Args:
            params_dict: Dictionary containing parameter updates.
        """
        pass


class EpsilonGreedy(ActionSelector):
    """
    Performs epsilon-greedy exploration.
    """

    def __init__(self, epsilon: float = DEFAULT_EPSILON):
        """
        Initializes the EpsilonGreedy selector.

        Args:
            epsilon: Probability of taking a random action.
        """
        self.epsilon = epsilon

    def select(self, values: torch.Tensor, exploration: bool = True) -> torch.Tensor:
        """
        Performs epsilon-greedy on values.

        Args:
            values: Q-values or logits.
            exploration: Whether to use epsilon-greedy exploration.

        Returns:
            Action tensor.
        """
        if exploration and np.random.rand() < self.epsilon:
            # Random action
            return torch.randint(
                0, values.shape[-1], values.shape[:-1], device=values.device
            )

        # Greedy action
        return values.argmax(dim=-1)

    def update_parameters(self, params: Dict[str, Any]) -> None:
        """
        Updates the epsilon parameter if present in params.
        """
        if "epsilon" in params:
            assert isinstance(
                params["epsilon"], (float, int)
            ), "Epsilon must be a number"
            self.epsilon = float(params["epsilon"])


class ArgmaxSelector(ActionSelector):
    """
    Selects the action with the highest value/logit.
    """

    def select(
        self, values: torch.Tensor, info: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Selects the action with the highest value.
        """
        # We can add masking logic here if needed, similar to the old implementation
        # But for now, let's keep it simple as the user didn't specify masking.
        # Actually, let's add it for compatibility.
        from utils.utils import get_legal_moves, action_mask

        q_values = values
        if info is not None and "legal_moves" in info:
            legal_moves = get_legal_moves(info)
            q_values = action_mask(
                q_values, legal_moves, mask_value=-float("inf"), device=q_values.device
            )

        return q_values.argmax(dim=-1)


class CategoricalSelector(ActionSelector):
    """
    Samples an action from a categorical distribution or logits.
    """

    def select(self, values: Any, **kwargs) -> torch.Tensor:
        """
        Samples an action.
        """
        if isinstance(values, torch.distributions.Distribution):
            return values.sample()

        # If logits:
        probs = torch.softmax(values, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)


class TemperatureSelector(ActionSelector):
    """
    Samples an action based on search policy and temperature using the power rule.
    Used by MuZero-style search agents.
    """

    def select(self, values: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Samples an action using the power rule: probs = values ** (1/temp).

        Args:
            values: Probabilities (e.g., visit counts normalized).
            temperature: Temperature for sampling. Higher values mean more exploration.

        Returns:
            Action tensor.
        """
        if temperature == 0:
            return values.argmax(dim=-1)

        assert temperature > 0, f"Temperature must be non-negative, got {temperature}"

        # Apply power rule: [B, A]
        # We divide by the max value for numerical stability before applying the power
        max_vals = values.max(dim=-1, keepdim=True).values
        # Avoid division by zero if all values are 0
        safe_values = values / (max_vals + 1e-10)
        probs = safe_values.pow(1.0 / temperature)

        # Normalize to get valid probabilities
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Sample: [B, A] -> [B, 1] -> [B]
        action = torch.multinomial(probs, 1).squeeze(-1)
        return action
