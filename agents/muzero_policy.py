import torch
import numpy as np
import time
from typing import Any, Dict, Optional
from search.search_factories import create_mcts
from agents.policy import Policy


class MuZeroPolicy(Policy):
    """
    MuZeroPolicy handles the MCTS search and network inference logic for MuZero.
    It implements the Policy interface to be used with a GenericActor.
    """

    def __init__(
        self,
        config,
        device,
        num_actions,
        observation_dimensions,
        model: Optional[Any] = None,
    ):
        self.config = config
        self.device = device
        self.num_actions = num_actions
        self.model = model
        self.search = create_mcts(config, device, self.num_actions)

        # Store observation dimensions for preprocessing
        self.observation_dimensions = observation_dimensions
        if not isinstance(self.observation_dimensions, torch.Size):
            self.observation_dimensions = torch.Size(self.observation_dimensions)

        self.last_prediction = {}
        self.steps_in_episode = 0

    def reset(self, state: Any) -> None:
        """Resets the policy state for a new episode."""
        self.steps_in_episode = 0
        self.last_prediction = {}

    def compute_action(self, obs: Any, info: Dict[str, Any] = None) -> Any:
        # 1. Determine current temperature
        curr_temp = self._get_current_temperature()

        # 2. Run prediction (MCTS)
        # Handle model selection modularly
        assert (
            self.model is not None
        ), "MuZeroPolicy requires a model for inference. Set it in __init__ or via self.model = model"

        prediction = self.predict(
            obs,
            info,
        )

        # 3. Select action
        action_tensor = self.select_actions(
            prediction,
            temperature=curr_temp,
        )

        # 4. Store metadata for get_info
        self.last_prediction = {
            "policy": prediction[1].detach(),
            "value": prediction[2],
            "best_action": prediction[3],
            "search_metadata": prediction[4],
        }

        self.steps_in_episode += 1
        return action_tensor

    def get_info(self) -> Dict[str, Any]:
        return self.last_prediction

    def _get_current_temperature(self) -> float:
        curr_temp = self.config.temperatures[0]
        for i, temperature_step in enumerate(self.config.temperature_updates):
            if not self.config.temperature_with_training_steps:
                if self.steps_in_episode >= temperature_step:
                    curr_temp = self.config.temperatures[i + 1]
                else:
                    break
        return curr_temp

    def preprocess(self, states) -> torch.Tensor:
        """
        Converts states to torch tensors on the correct device.
        Adds batch dimension if input is a single observation.
        """
        if torch.is_tensor(states):
            if states.device == self.device and states.dtype == torch.float32:
                # Already a tensor on the correct device and dtype
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

    def predict_initial_inference(self, states, model=None):
        if model is None:
            model = self.model
        state_inputs = self.preprocess(states)
        values, policies, hidden_states = model.initial_inference(state_inputs)
        return values, policies, hidden_states

    def predict_recurrent_inference(
        self,
        states,
        actions_or_codes,
        reward_h_states=None,
        reward_c_states=None,
        model=None,
    ):
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
        self, hidden_states, actions, model=None
    ):
        if model is None:
            model = self.model
        afterstates, value, chance_probs = model.afterstate_recurrent_inference(
            hidden_states,
            actions,
        )
        return afterstates, value, chance_probs

    def predict(
        self,
        state,
        info: Dict[str, Any] = None,
    ):
        # We need to handle multi-player 'to_play' logic
        # This originally used 'env' which we don't have here directly in a clean way
        # unless it's passed in info.

        to_play = 0
        if info and "player" in info:
            to_play = info["player"]
        elif info and "agent_selection" in info:
            # Fallback for PettingZoo style info
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

        return (
            exploratory_policy,
            target_policy,
            root_value,
            best_action,
            search_metadata,
        )

    def select_actions(
        self,
        prediction,
        temperature=0.0,
    ):
        if temperature != 0:
            probs = prediction[0] ** (1 / temperature)
            probs /= probs.sum()
            action = torch.multinomial(probs, 1)
            return action
        else:
            return prediction[3]
