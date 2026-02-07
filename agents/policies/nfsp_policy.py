from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import random
from agents.policies.policy import Policy
from agents.action_selectors.selectors import ActionSelector


class NFSPPolicy(Policy):
    """
    Policy for Neural Fictitious Self-Play (NFSP).
    Switches between a Best Response policy and an Average Strategy policy
    based on the anticipatory parameter (eta).
    """

    def __init__(
        self,
        best_response_model: nn.Module,
        average_model: nn.Module,
        best_response_selector: ActionSelector,
        average_selector: ActionSelector,
        device: torch.device = torch.device("cpu"),
        eta: float = 0.1,
    ):
        """
        Initializes the NFSPPolicy.

        Args:
            best_response_model: Network for the Best Response policy (often DQN).
            average_model: Network for the Average Strategy policy (supervised imitation).
            best_response_selector: Selector for action selection in Best Response.
            average_selector: Selector for action selection in Average Strategy.
            device: Torch device.
            eta: Anticipatory parameter. Probability of using the Best Response policy.
        """
        self.best_response_model = best_response_model
        self.average_model = average_model
        self.best_response_selector = best_response_selector
        self.average_selector = average_selector
        self.device = device
        self.eta = eta

        self.best_response_model.to(self.device).eval()
        self.average_model.to(self.device).eval()

        self.current_policy = "average_strategy"

    def reset(self, state: Any = None) -> None:
        """
        Resets the policy state for a new episode.
        Decides whether to use Best Response or Average Strategy for the entire episode.
        """
        if random.random() < self.eta:
            self.current_policy = "best_response"
        else:
            self.current_policy = "average_strategy"

        if hasattr(self.best_response_model, "reset_noise") and callable(
            self.best_response_model.reset_noise
        ):
            self.best_response_model.reset_noise()
        if hasattr(self.average_model, "reset_noise") and callable(
            self.average_model.reset_noise
        ):
            self.average_model.reset_noise()

    def compute_action(self, obs: Any, info: Dict[str, Any] = None) -> Any:
        """
        Computes an action given an observation and info.
        Uses the policy decided at the start of the episode.
        """
        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.tensor(
                obs, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
        else:
            obs_tensor = obs.to(self.device)
            # Ensure batch dimension
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

        with torch.inference_mode():
            if self.current_policy == "best_response":
                predictions = self.best_response_model(obs_tensor)
                selector = self.best_response_selector
            else:
                predictions = self.average_model(obs_tensor)
                selector = self.average_selector

        action = selector.select(predictions, info=info)

        # Squeeze out batch dimension if we had a single observation
        if (
            isinstance(action, torch.Tensor)
            and action.dim() > 0
            and action.shape[0] == 1
        ):
            action = action.squeeze(0)

        return action

    def get_info(self) -> Dict[str, Any]:
        """
        Returns metadata about the last decision.
        """
        return {"policy": self.current_policy}

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        """
        Updates the internal parameters of the policy and its selectors.
        """
        if "eta" in params_dict:
            self.eta = float(params_dict["eta"])

        # Forward parameters to selectors
        if self.best_response_selector is not None:
            self.best_response_selector.update_parameters(params_dict)
        if self.average_selector is not None:
            self.average_selector.update_parameters(params_dict)

        # Optional: Update model weights if provided
        if "best_response_state_dict" in params_dict:
            self.best_response_model.load_state_dict(
                params_dict["best_response_state_dict"]
            )
        if "average_state_dict" in params_dict:
            self.average_model.load_state_dict(params_dict["average_state_dict"])
