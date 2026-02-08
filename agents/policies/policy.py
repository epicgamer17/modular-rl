from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, Optional


class Policy(ABC):
    """
    Abstract interface for policies.
    """

    @abstractmethod
    def reset(self, state: Any) -> None:
        """
        Resets the policy state for a new episode.
        """
        pass

    @abstractmethod
    def compute_action(self, obs: Any, info: Dict[str, Any] = None, **kwargs) -> Any:
        """
        Computes an action given an observation and info.
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Returns metadata about the last decision (e.g., policy, value).
        """
        return {}

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        """
        Updates the internal parameters of the policy.
        """
        pass
