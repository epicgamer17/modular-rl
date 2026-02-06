import torch
import numpy as np
import time
from search.search_factories import create_mcts
from replay_buffers.game import Game


class MuZeroActor:
    """
    MuZeroActor handles the game-playing logic, including MCTS search and inference.
    It is designed to be purely functional and stateless regarding training.
    """

    def __init__(self, config, device, num_actions, observation_dimensions):
        self.config = config
        self.device = device
        self.num_actions = num_actions
        self.search = create_mcts(config, device, self.num_actions)
        # Store observation dimensions for preprocessing
        self.observation_dimensions = observation_dimensions
        if not isinstance(self.observation_dimensions, torch.Size):
            self.observation_dimensions = torch.Size(self.observation_dimensions)

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

    def predict_initial_inference(self, states, model):
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

    def predict_afterstate_recurrent_inference(self, hidden_states, actions, model):
        afterstates, value, chance_probs = model.afterstate_recurrent_inference(
            hidden_states,
            actions,
        )
        return afterstates, value, chance_probs

    def predict(
        self,
        state,
        info: dict = None,
        env=None,
        inference_model=None,
    ):
        if self.config.game.num_players != 1:
            # Assumes env follows PettingZoo AEC indexing or similar
            if hasattr(env, "agents") and hasattr(env, "agent_selection"):
                to_play = env.agents.index(env.agent_selection)
            else:
                to_play = info.get("player", 0)
        else:
            to_play = 0

        inference_fns = {
            "initial": self.predict_initial_inference,
            "recurrent": self.predict_recurrent_inference,
            "afterstate": self.predict_afterstate_recurrent_inference,
        }

        root_value, exploratory_policy, target_policy, best_action, search_metadata = (
            self.search.run(
                state, info, to_play, inference_fns, inference_model=inference_model
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

    def play_game(self, env, model, temperature=None, stats_tracker=None):
        """
        Runs one episode and returns the Game object.
        """
        start_time = time.time()
        with torch.no_grad():
            if self.config.game.num_players != 1:
                env.reset()
                state, reward, terminated, truncated, info = env.last()
                agent_id = env.agent_selection
                current_player = env.agents.index(agent_id)
            else:
                state, info = env.reset()

            game: Game = Game(self.config.game.num_players)
            game.append(state, info)

            done = False
            while not done:
                # Temperature logic
                if temperature is None:
                    curr_temp = self.config.temperatures[0]
                    # Note: training_step based temperature updates might be tricky if actor is stateless
                    # For now, we use length of game if not training_step_with_temperature
                    for i, temperature_step in enumerate(
                        self.config.temperature_updates
                    ):
                        if not self.config.temperature_with_training_steps:
                            if len(game) >= temperature_step:
                                curr_temp = self.config.temperatures[i + 1]
                            else:
                                break
                        else:
                            # If we need training_step, it must be passed in or managed elsewhere
                            # Defaulting to first temperature if we can't determine
                            pass
                else:
                    curr_temp = temperature

                prediction = self.predict(
                    state,
                    info,
                    env=env,
                    inference_model=model,
                )

                action_tensor = self.select_actions(
                    prediction,
                    temperature=curr_temp,
                )
                action = action_tensor.item()

                if self.config.game.num_players != 1:
                    env.step(action)
                    next_state, _, terminated, truncated, next_info = env.last()
                    reward = env.rewards[env.agents[current_player]]
                    agent_id = env.agent_selection
                    current_player = env.agents.index(agent_id)
                else:
                    next_state, reward, terminated, truncated, next_info = env.step(
                        action
                    )

                done = terminated or truncated

                game.append(
                    observation=next_state,
                    info=next_info,
                    action=action,
                    reward=reward,
                    policy=prediction[1].detach(),
                    value=prediction[2],
                )

                # if stats_tracker:
                #     self._track_search_stats(prediction[4], stats_tracker)

                state = next_state
                info = next_info

            if stats_tracker:
                duration = time.time() - start_time
                if duration > 0:
                    fps = len(game) / duration
                    stats_tracker.append("actor_fps", fps)

            return game

    def _track_search_stats(self, search_metadata, stats_tracker):
        """Track statistics from the search process."""
        if search_metadata is None or stats_tracker is None:
            return

        network_policy = search_metadata["network_policy"]
        search_policy = search_metadata["search_policy"]
        network_value = search_metadata["network_value"]
        search_value = search_metadata["search_value"]

        # Policy Entropy
        probs = search_policy + 1e-10
        entropy = -torch.sum(probs * torch.log(probs)).item()
        stats_tracker.append("policy_entropy", entropy)

        # Value Difference
        stats_tracker.append("value_diff", abs(search_value - network_value))

        # Policy Improvement
        stats_tracker.append(
            "policy_improvement", network_policy.detach().unsqueeze(0), subkey="network"
        )
        stats_tracker.append(
            "policy_improvement", search_policy.detach().unsqueeze(0), subkey="search"
        )
