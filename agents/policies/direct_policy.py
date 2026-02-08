from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from agents.policies.policy import Policy
from agents.action_selectors.selectors import ActionSelector


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
        support: Optional[torch.Tensor] = None,
    ):
        """
        Initializes the DirectPolicy.

        Args:
            model: Neural network for inference.
            action_selector: Selector for choosing actions from values.
            device: Torch device.
            support: Optional support tensor for distributional RL (C51).
                     If provided, distributions are converted to Q-values
                     before action selection.
        """
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.support = support.to(device) if support is not None else None

        self.model.to(self.device)
        self.model.eval()

    def reset(self, state: Any = None) -> None:
        """
        Resets the policy state. This resets noise in the model if applicable.
        """
        if hasattr(self.model, "reset_noise"):
            self.model.reset_noise()

    def compute_action(self, obs: Any, info: Dict[str, Any] = None, **kwargs) -> Any:
        """
        Computes an action given an observation and info.
        """
        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.tensor(
                obs, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
        else:
            obs_tensor = obs.to(self.device)
            if (
                obs_tensor.dim() == len(self.model.input_shape) - 1
            ):  # Assumes model.input_shape includes batch
                obs_tensor = obs_tensor.unsqueeze(0)

        with torch.inference_mode():
            predictions = self.model(obs_tensor)

        # For distributional RL (C51): convert distributions to Q-values
        # predictions shape: (B, num_actions, atoms)
        # Q-value = sum(distribution * support) for each action
        if self.support is not None and predictions.dim() == 3:
            q_values = (predictions * self.support).sum(dim=-1)  # (B, num_actions)
        else:
            q_values = predictions

        action = self.action_selector.select(q_values, info=info)

        # Squeeze out batch dimension if we had a single observation
        if action.dim() > 0 and action.shape[0] == 1:
            action = action.squeeze(0)

        return action

    def get_info(self) -> Dict[str, Any]:
        """
        Returns metadata about the last decision (e.g., policy, value).
        Currently placeholder.
        """
        return {}

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        """
        Updates the internal parameters of the policy and its selector.

        Args:
            params_dict: Dictionary containing parameter updates.
                         Selector parameters (e.g., 'epsilon') are forwarded to action_selector.
                         If 'model_state_dict' is present, updates the model weights.
        """
        # Forward parameters to the action selector
        if self.action_selector is not None:
            self.action_selector.update_parameters(params_dict)

        # Optional: Update model weights if provided
        if "model_state_dict" in params_dict:
            self.model.load_state_dict(params_dict["model_state_dict"])
