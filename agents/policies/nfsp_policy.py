"""
NFSPPolicy for Neural Fictitious Self-Play.
Supports per-player policy selection (BR vs AVG) and optional separate weights per player.
"""

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import random
from agents.policies.policy import Policy
from agents.action_selectors.selectors import BaseActionSelector as ActionSelector


class NFSPPolicy(Policy):
    """
    Policy for Neural Fictitious Self-Play (NFSP).

    Supports two modes:
    1. Shared weights: All players share the same BR/AVG networks (default)
    2. Separate weights: Each player has their own BR/AVG networks

    Each player independently samples whether to use BR or AVG at episode start.
    """

    def __init__(
        self,
        # Shared weights mode (single model for all players)
        best_response_model: Optional[nn.Module] = None,
        average_model: Optional[nn.Module] = None,
        # Separate weights mode (per-player models)
        best_response_models: Optional[Dict[str, nn.Module]] = None,
        average_models: Optional[Dict[str, nn.Module]] = None,
        # Selectors (shared for all players)
        best_response_selector: Optional[ActionSelector] = None,
        average_selector: Optional[ActionSelector] = None,
        device: torch.device = torch.device("cpu"),
        eta: float = 0.1,
        player_ids: Optional[List[str]] = None,
    ):
        """
        Initializes the NFSPPolicy.

        Args:
            best_response_model: Network for BR policy (shared mode).
            average_model: Network for AVG policy (shared mode).
            best_response_models: Dict mapping player_id -> BR network (separate mode).
            average_models: Dict mapping player_id -> AVG network (separate mode).
            best_response_selector: Selector for BR action selection.
            average_selector: Selector for AVG action selection.
            device: Torch device.
            eta: Anticipatory parameter. Probability of using BR policy.
            player_ids: List of player IDs for the game.
        """
        self.device = device
        self.eta = eta
        self.best_response_selector = best_response_selector
        self.average_selector = average_selector

        # Determine mode based on which models are provided
        if best_response_models is not None and average_models is not None:
            # Separate weights mode
            self.shared_weights = False
            self.br_models = best_response_models
            self.avg_models = average_models
            self.player_ids = list(best_response_models.keys())

            # Move all models to device
            for model in self.br_models.values():
                model.to(self.device).eval()
            for model in self.avg_models.values():
                model.to(self.device).eval()
        else:
            # Shared weights mode (backwards compatible)
            self.shared_weights = True
            assert best_response_model is not None, "Must provide best_response_model"
            assert average_model is not None, "Must provide average_model"
            self.br_model = best_response_model
            self.avg_model = average_model
            self.br_model.to(self.device).eval()
            self.avg_model.to(self.device).eval()
            self.player_ids = player_ids if player_ids else ["player_0"]

        # Per-player policy mode tracking
        self.current_policy: Dict[str, str] = {
            pid: "average_strategy" for pid in self.player_ids
        }

    def reset(self, state: Any = None) -> None:
        """
        Resets the policy state for a new episode.
        Each player independently decides BR vs AVG for this episode.
        """
        for player_id in self.player_ids:
            self.reset_player(player_id)

    def reset_player(self, player_id: str) -> None:
        """
        Resets policy for a specific player.
        Samples whether to use BR or AVG for this player's episode.

        Args:
            player_id: The player ID to reset.
        """
        if random.random() < self.eta:
            self.current_policy[player_id] = "best_response"
        else:
            self.current_policy[player_id] = "average_strategy"

        # Reset noise in models
        if self.shared_weights:
            if hasattr(self.br_model, "reset_noise"):
                self.br_model.reset_noise()
            if hasattr(self.avg_model, "reset_noise"):
                self.avg_model.reset_noise()
        else:
            br_model = self.br_models.get(player_id)
            avg_model = self.avg_models.get(player_id)
            if br_model and hasattr(br_model, "reset_noise"):
                br_model.reset_noise()
            if avg_model and hasattr(avg_model, "reset_noise"):
                avg_model.reset_noise()

    def compute_action(
        self,
        obs: Any,
        info: Dict[str, Any] = None,
        player_id: Optional[str] = None,
        exploration: bool = True,
    ) -> Any:
        """
        Computes an action given an observation and info.

        Args:
            obs: The observation.
            info: Additional info dict.
            player_id: The player making this action. Defaults to first player.
            exploration: Whether to sample (True) or act greedily (False).
                         For NFSP, we typically always want sampling to play
                         the mixed strategy properly.

        Returns:
            The selected action.
        """
        if player_id is None:
            player_id = self.player_ids[0]

        # Ensure player has a policy mode set
        if player_id not in self.current_policy:
            self.reset_player(player_id)

        # Prepare observation tensor
        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.tensor(
                obs, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
        else:
            obs_tensor = obs.to(self.device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

        # Get the appropriate model and selector
        policy_mode = self.current_policy[player_id]

        with torch.inference_mode():
            if policy_mode == "best_response":
                if self.shared_weights:
                    net_out = self.br_model.initial_inference(obs_tensor)
                else:
                    net_out = self.br_models[player_id].initial_inference(obs_tensor)
                selector = self.best_response_selector
            else:
                if self.shared_weights:
                    net_out = self.avg_model.initial_inference(obs_tensor)
                else:
                    net_out = self.avg_models[player_id].initial_inference(obs_tensor)
                selector = self.average_selector

        # Pass exploration flag to selector for proper sampling behavior
        # selector.select_action expects (agent_network, state, info, network_output, **kwargs)
        # We pass None as agent_network since we already computed key_output
        action, _ = selector.select_action(
            None, obs, info=info, network_output=net_out, exploration=exploration
        )

        # Squeeze out batch dimension if single observation
        if (
            isinstance(action, torch.Tensor)
            and action.dim() > 0
            and action.shape[0] == 1
        ):
            action = action.squeeze(0)

        return action

    def get_current_policy(self, player_id: Optional[str] = None) -> str:
        """
        Returns the current policy mode for a player.

        Args:
            player_id: The player to query. Defaults to first player.

        Returns:
            "best_response" or "average_strategy"
        """
        if player_id is None:
            player_id = self.player_ids[0]
        return self.current_policy.get(player_id)

    def get_info(self) -> Dict[str, Any]:
        """
        Returns metadata about the current state.
        """
        return {"policies": self.current_policy.copy()}

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        """
        Updates the internal parameters of the policy and its selectors.

        Args:
            params_dict: Dictionary containing parameter updates.
                - 'eta': Update anticipatory parameter
                - 'best_response_state_dict': Update shared BR model weights
                - 'average_state_dict': Update shared AVG model weights
                - 'best_response_state_dicts': Dict[player_id, state_dict] for separate mode
                - 'average_state_dicts': Dict[player_id, state_dict] for separate mode
        """
        if "eta" in params_dict:
            self.eta = float(params_dict["eta"])

        # Forward parameters to selectors
        if self.best_response_selector is not None:
            self.best_response_selector.update_parameters(params_dict)
        if self.average_selector is not None:
            self.average_selector.update_parameters(params_dict)

        # Update model weights
        if self.shared_weights:
            if "best_response_state_dict" in params_dict:
                self.br_model.load_state_dict(params_dict["best_response_state_dict"])
            if "average_state_dict" in params_dict:
                self.avg_model.load_state_dict(params_dict["average_state_dict"])
        else:
            # Separate weights mode
            if "best_response_state_dicts" in params_dict:
                for pid, state_dict in params_dict["best_response_state_dicts"].items():
                    if pid in self.br_models:
                        self.br_models[pid].load_state_dict(state_dict)
            if "average_state_dicts" in params_dict:
                for pid, state_dict in params_dict["average_state_dicts"].items():
                    if pid in self.avg_models:
                        self.avg_models[pid].load_state_dict(state_dict)
