from typing import Any, Dict, Optional, Tuple
import torch
from agents.action_selectors.selectors import BaseActionSelector
from torch.distributions import Categorical


class PPODecorator(BaseActionSelector):
    """
    Decorator that injects PPO-specific metadata (log_prob, value)
    into the selection result.
    """

    def __init__(self, inner_selector: BaseActionSelector):
        self.inner_selector = inner_selector

    def select_action(
        self,
        agent_network: torch.nn.Module,
        obs: Any,
        info: Optional[Dict[str, Any]] = None,
        network_output: Optional[Any] = None,
        exploration: Optional[bool] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # 1. Ensure we have network output
        if network_output is None:
            network_output = agent_network.initial_inference(obs)

        # 2. Delegate selection to the inner selector
        action, metadata = self.inner_selector.select_action(
            agent_network,
            obs,
            info,
            network_output=network_output,
            exploration=exploration,
            **kwargs
        )

        # 3. Inject PPO metadata
        # We assume network_output.policy is a Distribution or has log_prob
        if hasattr(network_output.policy, "log_prob"):
            # PPO usually needs log_prob of the chosen action
            metadata["log_prob"] = network_output.policy.log_prob(action).detach().cpu()
        if hasattr(network_output, "value"):
            metadata["value"] = network_output.value.detach().cpu()

        return action, metadata

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        """
        Pass parameter updates down to the inner selector.
        """
        if hasattr(self.inner_selector, "update_parameters"):
            self.inner_selector.update_parameters(params_dict)


class MCTSDecorator(BaseActionSelector):
    """
    Decorator that runs MCTS before selecting an action.
    It wraps a selector (usually TemperatureSelector) that chooses from the MCTS policies.
    """

    def __init__(
        self, inner_selector: BaseActionSelector, search_algorithm: Any, config: Any
    ):
        self.inner_selector = inner_selector
        self.search = search_algorithm
        self.config = config  # Need config for temperature schedule

    def _get_current_temperature(self, steps_in_episode: int) -> float:
        """Determines exploration temperature based on episode step."""
        if not hasattr(self.config, "temperatures") or not hasattr(
            self.config, "temperature_updates"
        ):
            return 1.0  # Default if no config

        curr_temp = self.config.temperatures[0]
        for i, temperature_step in enumerate(self.config.temperature_updates):
            if not getattr(self.config, "temperature_with_training_steps", False):
                if steps_in_episode >= temperature_step:
                    curr_temp = self.config.temperatures[i + 1]
                else:
                    break
        return curr_temp

    def select_action(
        self,
        agent_network: torch.nn.Module,
        obs: Any,
        info: Optional[Dict[str, Any]] = None,
        network_output: Optional[
            Any
        ] = None,  # Not used for MCTS usually, as MCTS does its own inference calls
        exploration: Optional[bool] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # 1. Determine "to_play"
        to_play = 0
        if info and "player" in info:
            to_play = info["player"]
        elif info and "agent_selection" in info:
            to_play = info.get("to_play", 0)

        # Get episode step for temperature
        # If exploration is explicitly False, we might want to force temperature to something low?
        # But really, if exploration=False, the inner selector (Categorical) will take Argmax, so temperature matters less
        # unless it changes the mode.
        episode_step = kwargs.get("episode_step", 0)
        curr_temp = self._get_current_temperature(episode_step)

        # 2. Define inference wrapper functions that bind the model
        def predict_initial_inference(states):
            return agent_network.initial_inference(states)

        def predict_recurrent_inference(network_state, actions):
            return agent_network.recurrent_inference(network_state, actions)

        # Handle stochastic MuZero if applicable
        def predict_afterstate_recurrent_inference(hidden_states, actions):
            return agent_network.afterstate_recurrent_inference(hidden_states, actions)

        inference_fns = {
            "initial": predict_initial_inference,
            "recurrent": predict_recurrent_inference,
            "afterstate": predict_afterstate_recurrent_inference,
        }

        # 3. Run MCTS
        # Search algorithms usually expect: run(state, info, to_play, inference_fns, inference_model)
        root_value, exploratory_policy, target_policy, best_action, search_metadata = (
            self.search.run(
                obs, info, to_play, inference_fns, inference_model=agent_network
            )
        )

        # 4. Context for inner selector (e.g. TemperatureSelector)
        # Inner selector chooses from 'exploratory_policy' (counts/probs)
        # We need to adapt the interface: BaseActionSelector.select_action usually takes 'agent_network' and 'obs'
        # But here we want it to select from 'exploratory_policy'.
        #
        # Interface mismatch! BaseActionSelector expects (network, obs).
        # TemperatureSelector and CategoricalSelector usually work on Distributions from network.
        # Here we have a Distribution (exploratory_policy) explicitly.
        #
        # OPTION A: Create a dummy InferenceOutput wrapping the policy and pass it to inner.
        # OPTION B: Inner selector knows how to handle 'policy' kwarg.

        # We will use Option A/B hybrid: Pass 'network_output' with policy set to exploratory_policy.
        from modules.world_models.inference_output import InferenceOutput

        # Mocking a distribution object if exploratory_policy is just probs?
        # Typically MCTS 'exploratory_policy' is a tensor of probs or counts.
        # If it's counts, we might need to normalize or let selector handle it.
        # Assuming normalized probs for now or selector handles it.

        # Apply temperature to probabilities
        # exploratory_policy is a tensor of probabilities (from MCTS)
        if curr_temp != 1.0 and curr_temp > 0:
            # Avoid numerical instability with log
            log_probs = torch.log(exploratory_policy + 1e-8)
            log_probs = log_probs / curr_temp
            heated_probs = torch.softmax(log_probs, dim=-1)
            mcts_policy_dist = Categorical(probs=heated_probs)
        else:
            mcts_policy_dist = Categorical(probs=exploratory_policy)

        # Use a dummy wrapper or struct
        class DictOutput:
            def __init__(self, policy):
                self.policy = policy

        # We can construct a InferenceOutput-like object
        # MCTS returns probs usually.
        mcts_output = InferenceOutput(
            value=torch.tensor(root_value), policy=mcts_policy_dist
        )

        # Delegate to Inner Selector
        # Note: We pass kwargs (like temperature) through
        # Update kwargs with temperature
        kwargs["temperature"] = curr_temp

        action, _ = self.inner_selector.select_action(
            agent_network,
            obs,
            info,
            network_output=mcts_output,
            exploration=exploration,
            **kwargs
        )

        # 5. Metadata
        # Helper to detach
        def detach_all(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu()
            elif isinstance(obj, dict):
                return {k: detach_all(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [detach_all(v) for v in obj]
            return obj

        metadata = {
            "policy": (
                target_policy.detach().cpu()
                if isinstance(target_policy, torch.Tensor)
                else target_policy
            ),
            "value": float(root_value),
            "best_action": int(action),
            "search_metadata": detach_all(search_metadata),
            "root_value": float(root_value),
        }

        return action, metadata

    # Removed reset() - Stateless

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        if hasattr(self.inner_selector, "update_parameters"):
            self.inner_selector.update_parameters(params_dict)
