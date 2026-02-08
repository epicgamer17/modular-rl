import torch
import numpy as np
import time
from typing import Any, Dict, Optional, Protocol, Union
from agents.policies.policy import Policy
from agents.action_selectors.selectors import ActionSelector, TemperatureSelector


class SearchAlgorithmProtocol(Protocol):
    """Protocol for search algorithms to enable duck typing."""

    def run(
        self,
        state: Any,
        info: Dict[str, Any],
        to_play: int,
        inference_fns: Dict[str, Any],
        inference_model: Any = None,
    ) -> tuple:
        """Runs the search and returns results."""
        ...


class SearchPolicy(Policy):
    """
    SearchPolicy handles the MCTS search and network inference logic.
    It implements a Policy interface to be used with a BaseActor.

    This class uses dependency injection for model and search_algorithm,
    making it reusable across different MuZero-style agents.
    """

    def __init__(
        self,
        model: Any,
        search_algorithm: SearchAlgorithmProtocol,
        config: Any,
        device: torch.device,
        observation_dimensions: tuple,
        action_selector: Optional[ActionSelector] = None,
    ):
        """
        Initializes SearchPolicy with injected dependencies.

        Args:
            model: Neural network model for inference (must support initial_inference,
                   recurrent_inference, afterstate_recurrent_inference).
            search_algorithm: Search algorithm instance (e.g., MCTS).
            config: Configuration object containing policy parameters.
            device: Torch device for tensor operations.
            observation_dimensions: Shape of a single observation (without batch dim).
            action_selector: Optional action selector. Defaults to TemperatureSoftmax.
        """
        self.model = model
        self.search = search_algorithm
        self.action_selector = action_selector or TemperatureSelector()
        self.config = config
        self.device = device

        # Store observation dimensions for preprocessing
        if isinstance(observation_dimensions, torch.Size):
            self.observation_dimensions = observation_dimensions
        else:
            self.observation_dimensions = torch.Size(observation_dimensions)

        self.last_prediction: Dict[str, Any] = {}
        self.steps_in_episode = 0

    def reset(self, state: Any) -> None:
        """Resets the policy state for a new episode."""
        self.steps_in_episode = 0
        self.last_prediction = {}

    def compute_action(self, obs: Any, info: Dict[str, Any] = None, **kwargs) -> Any:
        """
        Computes an action given an observation.

        Args:
            obs: Current observation.
            info: Additional info dict (e.g., legal moves, player).
            **kwargs: Additional keyword arguments (e.g., player_id).

        Returns:
            Action tensor.
        """
        # 1. Determine current temperature
        curr_temp = self._get_current_temperature()

        # 2. Run prediction (MCTS)
        assert self.model is not None, "SearchPolicy requires a model for inference."

        action, policy_info = self.predict(obs, info)

        # 3. Select action (delegated to action_selector)
        # We use exploratory_policy (visit counts/probs) for sampling
        action_tensor = self.action_selector.select(
            policy_info["exploratory_policy"], temperature=curr_temp
        )

        # 4. Store metadata for get_info
        self.last_prediction = {
            "policy": policy_info["target_policy"].detach(),
            "value": policy_info["root_value"],
            "best_action": action,
            "search_metadata": policy_info["search_metadata"],
        }

        self.steps_in_episode += 1
        return action_tensor

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        """
        Updates the internal parameters of the policy and its selector.
        """
        if self.action_selector is not None:
            self.action_selector.update_parameters(params_dict)

    def get_info(self) -> Dict[str, Any]:
        """Returns metadata about the last decision."""
        return self.last_prediction

    def _get_current_temperature(self) -> float:
        """Determines exploration temperature based on episode step."""
        curr_temp = self.config.temperatures[0]
        for i, temperature_step in enumerate(self.config.temperature_updates):
            if not self.config.temperature_with_training_steps:
                if self.steps_in_episode >= temperature_step:
                    curr_temp = self.config.temperatures[i + 1]
                else:
                    break
        return curr_temp

    def preprocess(self, states: Any) -> torch.Tensor:
        """
        Converts states to torch tensors on the correct device.
        Adds batch dimension if input is a single observation.
        """
        if torch.is_tensor(states):
            if states.device == self.device and states.dtype == torch.float32:
                prepared_state = states
            else:
                prepared_state = states.to(self.device, dtype=torch.float32)
        else:
            np_states = np.array(states, copy=False)
            prepared_state = torch.tensor(
                np_states, dtype=torch.float32, device=self.device
            )

        if prepared_state.ndim == 0:
            prepared_state = prepared_state.unsqueeze(0)

        # states might be a single observation without batch dim
        if prepared_state.shape == self.observation_dimensions:
            prepared_state = prepared_state.unsqueeze(0)

        return prepared_state

    def predict_initial_inference(self, states: Any, model: Any = None) -> tuple:
        """Runs initial inference through the model."""
        if model is None:
            model = self.model
        state_inputs = self.preprocess(states)
        values, policies, hidden_states = model.initial_inference(state_inputs)
        return values, policies, hidden_states

    def predict_recurrent_inference(
        self,
        states: Any,
        actions_or_codes: Any,
        reward_h_states: Any = None,
        reward_c_states: Any = None,
        model: Any = None,
    ) -> tuple:
        """Runs recurrent inference through the model."""
        if model is None:
            model = self.model
        rewards, states, values, policies, to_play, reward_hidden = (
            model.recurrent_inference(
                states,
                actions_or_codes,
                reward_h_states,
                reward_c_states,
            )
        )

        reward_h_states = reward_hidden[0]
        reward_c_states = reward_hidden[1]

        return (
            rewards,
            states,
            values,
            policies,
            to_play,
            reward_h_states,
            reward_c_states,
        )

    def predict_afterstate_recurrent_inference(
        self, hidden_states: Any, actions: Any, model: Any = None
    ) -> tuple:
        """Runs afterstate recurrent inference for stochastic MuZero."""
        if model is None:
            model = self.model
        afterstates, value, chance_probs = model.afterstate_recurrent_inference(
            hidden_states, actions
        )
        return afterstates, value, chance_probs

    def predict(self, state: Any, info: Dict[str, Any] = None, **kwargs) -> tuple:
        """
        Runs the search algorithm to get action predictions.

        Delegates to self.search.run() with inference functions.

        Args:
            state: Current observation.
            info: Additional info dict (e.g., legal moves, player info).

        Returns:
            Tuple of (exploratory_policy, target_policy, root_value,
                      best_action, search_metadata).
        """
        # Handle multi-player 'to_play' logic
        to_play = 0
        if info and "player" in info:
            to_play = info["player"]
        elif info and "agent_selection" in info:
            to_play = info.get("to_play", 0)

        inference_fns = {
            "initial": self.predict_initial_inference,
            "recurrent": self.predict_recurrent_inference,
            "afterstate": self.predict_afterstate_recurrent_inference,
        }

        root_value, exploratory_policy, target_policy, best_action, search_metadata = (
            self.search.run(
                state, info, to_play, inference_fns, inference_model=self.model
            )
        )

        # Return standard (action, info_dict) format
        info_dict = {
            "exploratory_policy": exploratory_policy,
            "target_policy": target_policy,
            "root_value": root_value,
            "search_metadata": search_metadata,
        }

        return best_action, info_dict
