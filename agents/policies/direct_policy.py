from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from agents.policies.policy import Policy
from agents.action_selectors.action_selectors import ActionSelector


class DirectPolicy(Policy):
    """
    Standard Policy implementation for non-search agents (Rainbow, PPO, etc.).
    It runs model inference and uses an ActionSelector to choose an action.
    """

    def __init__(
        self,
        model: nn.Module,
        action_selector: ActionSelector,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.action_selector = action_selector
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    def reset(self, state: Any = None) -> None:
        """
        Resets the policy state. This resets noise in the model if applicable.
        """
        if hasattr(self.model, "reset_noise"):
            self.model.reset_noise()

        # If there are RNN states or other internal states, they should be handled here
        # or managed by the model itself during forward pass if preferred.

    def compute_action(self, obs: Any, info: Dict[str, Any] = None) -> Any:
        """
        Computes an action given an observation and info.
        """
        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.tensor(obs, device=self.device).unsqueeze(0)
        else:
            obs_tensor = obs.to(self.device)
            if (
                obs_tensor.dim() == len(self.model.input_shape) - 1
            ):  # Assumes model.input_shape includes batch
                obs_tensor = obs_tensor.unsqueeze(0)

        with torch.inference_mode():
            predictions = self.model(obs_tensor)

        action = self.action_selector.select_action(predictions, info)
        return action

    def get_info(self) -> Dict[str, Any]:
        """
        Returns metadata about the last decision (e.g., policy, value).
        Currently placeholder.
        """
        return {}
