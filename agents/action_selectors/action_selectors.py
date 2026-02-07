from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import torch
import numpy as np
from utils.utils import get_legal_moves, action_mask, epsilon_greedy_policy


class ActionSelector(ABC):
    """
    Abstract interface for action selection logic.
    """

    @abstractmethod
    def select_action(
        self, predictions: Any, info: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Selects an action based on model predictions and environment info.
        """
        pass


class ArgmaxSelector(ActionSelector):
    """
    Selects the action with the highest value/logit.
    """

    def select_action(
        self, predictions: torch.Tensor, info: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        q_values = predictions
        if info is not None and "legal_moves" in info:
            legal_moves = get_legal_moves(info)
            q_values = action_mask(
                q_values, legal_moves, mask_value=-float("inf"), device=q_values.device
            )

        # Simple argmax. Ties are handled by torch.argmax picking the first occurrence.
        # If we need random tie-breaking, we can implement it later.
        # TODO: Random tie breaking
        selected_actions = q_values.argmax(dim=-1)
        return selected_actions


class CategoricalSelector(ActionSelector):
    """
    Samples an action from a categorical distribution (e.g., for PPO).
    """

    def select_action(
        self, predictions: Any, info: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        # predictions is expected to be a torch.distributions.Distribution
        if isinstance(predictions, tuple):
            distribution = predictions[0]
        else:
            distribution = predictions

        selected_action = distribution.sample()
        return selected_action


class EpsilonGreedySelector(ActionSelector):
    """
    Wraps another selector with epsilon-greedy exploration.
    """

    def __init__(self, selector: ActionSelector, epsilon: float):
        self.selector = selector
        self.epsilon = epsilon

    def select_action(
        self, predictions: Any, info: Optional[Dict[str, Any]] = None
    ) -> Any:
        return epsilon_greedy_policy(
            predictions,
            info,
            self.epsilon,
            wrapper=lambda p, i: self.selector.select_action(p, i),
        )


class TemperatureSelector(ActionSelector):
    """
    Samples an action based on search policy and temperature.
    Used by MuZero-style search agents.
    """

    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature

    def select_action(
        self, predictions: tuple, info: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Selects an action based on temperature.

        Args:
            predictions: Tuple of (best_action, info_dict) from search.
            info: (Unused) Additional environment info.
        """
        best_action, info_dict = predictions

        if self.temperature != 0:
            # predictions[1] is info_dict which contains 'exploratory_policy'
            policy = info_dict["exploratory_policy"]
            probs = policy ** (1 / self.temperature)
            probs /= probs.sum()
            action = torch.multinomial(probs, 1)
            return action
        else:
            return best_action
