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

    def select(
        self,
        values: torch.Tensor,
        exploration: bool = True,
        info: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Performs epsilon-greedy on values.

        Args:
            values: Q-values or logits.
            exploration: Whether to use epsilon-greedy exploration.
            info: Optional dict with 'legal_moves' for action masking.

        Returns:
            Action tensor.
        """
        from utils.utils import action_mask

        # Check for valid legal moves (non-empty, non-None)
        legal_moves = None
        if info is not None and "legal_moves" in info:
            moves = info.get("legal_moves")
            if moves is not None and len(moves) > 0:
                legal_moves = moves

        # Apply action masking if legal moves provided
        if legal_moves is not None:
            if exploration and np.random.rand() < self.epsilon:
                # Random action from legal moves only
                return torch.tensor(np.random.choice(legal_moves), device=values.device)
            # Greedy with masking
            masked_values = action_mask(
                values, legal_moves, mask_value=-float("inf"), device=values.device
            )
            return masked_values.argmax(dim=-1).squeeze()

        # No legal move constraints - standard epsilon-greedy
        if exploration and np.random.rand() < self.epsilon:
            # Random action - scalar for single obs
            return torch.randint(0, values.shape[-1], (), device=values.device)

        # Greedy action
        return values.argmax(dim=-1).squeeze()

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
        self,
        values: torch.Tensor,
        exploration: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Selects the action with the highest value.

        Args:
            values: Q-values or logits tensor.
            exploration: Ignored for argmax (always greedy).
            info: Optional dict with 'legal_moves' for action masking.
        """
        from utils.utils import action_mask

        q_values = values

        # Only apply masking if we have valid legal_moves
        if info is not None and "legal_moves" in info:
            legal_moves = info.get("legal_moves")
            # Handle various cases where legal_moves is None, empty, or not constraining
            if legal_moves is not None and len(legal_moves) > 0:
                q_values = action_mask(
                    q_values,
                    legal_moves,
                    mask_value=-float("inf"),
                    device=q_values.device,
                )
            # If legal_moves is empty or None, just use all actions (no masking needed)

        return q_values.argmax(dim=-1).squeeze()


class CategoricalSelector(ActionSelector):
    """
    Samples an action from a categorical distribution, logits, or probabilities.
    """

    def __init__(self, from_logits: bool = True):
        """
        Initializes the CategoricalSelector.

        Args:
            from_logits: Whether the input values are raw logits (True) or
                         probabilities that sum to 1 (False).
        """
        self.from_logits = from_logits

    def select(
        self,
        values: Any,
        exploration: bool = True,
        info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Samples or selects the best action.

        Args:
            values: Model output (logits or probabilities).
            exploration: Whether to sample (True) or take argmax (False).
            info: Optional dict for action masking.
        """
        if isinstance(values, torch.distributions.Distribution):
            return (
                values.sample() if exploration else torch.argmax(values.probs, dim=-1)
            )

        # Handle tensor input
        if self.from_logits:
            probs = torch.softmax(values, dim=-1)
        else:
            probs = values

        # Handle Action Masking if provided
        if info is not None and "legal_moves" in info:
            from utils.utils import action_mask

            legal_moves = info.get("legal_moves")
            if legal_moves is not None and len(legal_moves) > 0:
                # Masking usually done on logits.
                # If we have probs, we set illegal to 0.0
                if self.from_logits:
                    values = action_mask(
                        values,
                        legal_moves,
                        mask_value=-float("inf"),
                        device=values.device,
                    )
                    probs = torch.softmax(values, dim=-1)
                else:
                    probs = action_mask(
                        probs, legal_moves, mask_value=0.0, device=probs.device
                    )
                    # Re-normalize if we masked probs
                    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

        if not exploration:
            return probs.argmax(dim=-1).squeeze()

        # Ensure probs for multinomial (B, num_actions)
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
            return torch.multinomial(probs, 1).squeeze()

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

    def update_parameters(self, params: Dict[str, Any]) -> None:
        """
        Updates the temperature parameter if present in params.
        """
        if "temperature" in params:
            assert isinstance(
                params["temperature"], (float, int)
            ), "Temperature must be a number"
            self.temperature = float(params["temperature"])
