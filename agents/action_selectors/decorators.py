import time
from typing import Any, Dict, Optional, Tuple
import torch
from agents.action_selectors.selectors import BaseActionSelector
from torch.distributions import Categorical
from utils.schedule import create_schedule, Schedule


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
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # 1. Ensure we have network output
        if network_output is None:
            network_output = agent_network.obs_inference(obs)

        # 2. Delegate selection to the inner selector
        action, metadata = self.inner_selector.select_action(
            agent_network,
            obs,
            info,
            network_output=network_output,
            exploration=exploration,
            **kwargs,
        )

        # 3. Inject PPO metadata
        # Use the (potentially masked) distribution from metadata if available,
        # otherwise fallback to the one in network_output.
        dist = metadata.get("dist", network_output.policy)
        if dist is not None:
            metadata["log_prob"] = dist.log_prob(action).cpu()
        metadata["value"] = network_output.value.cpu()

        return action, metadata

    def mask_actions(
        self,
        values: torch.Tensor,
        legal_moves: Any,
        mask_value: float = -float("inf"),
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return self.inner_selector.mask_actions(values, legal_moves, mask_value, device)

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        """
        Pass parameter updates down to the inner selector.
        """
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
        self.config = config
        if hasattr(config, "temperature_schedule"):
            self.temperature_schedule = create_schedule(config.temperature_schedule)
        else:
            from utils.schedule import ScheduleConfig

            self.temperature_schedule = create_schedule(
                ScheduleConfig.stepwise(steps=[5], values=[1.0, 0.0])
            )
        # Cache for temperature schedule to avoid O(N^2)
        self._last_step = -1

    def _get_current_temperature(self, steps_in_episode: int) -> float:
        """Determines exploration temperature based on episode step."""
        if not self.temperature_schedule.config.with_training_steps:
            # If we haven't reached this step yet, we can't step forward blindly
            # but usually steps_in_episode increases monotonically.
            if steps_in_episode > self._last_step:
                for _ in range(steps_in_episode - self._last_step):
                    self.temperature_schedule.step()
                self._last_step = steps_in_episode
            elif steps_in_episode < self._last_step:
                # Fallback: recreate if we go backwards (e.g. new episode)
                self.temperature_schedule = create_schedule(
                    self.config.temperature_schedule
                )
                for _ in range(steps_in_episode):
                    self.temperature_schedule.step()
                self._last_step = steps_in_episode
            return self.temperature_schedule.get_value()
        return self.temperature_schedule.get_value()

    def select_action(
        self,
        agent_network: torch.nn.Module,
        obs: Any,
        info: Optional[Dict[str, Any]] = None,
        network_output: Optional[
            Any
        ] = None,  # Not used for MCTS usually, as MCTS does its own inference calls
        exploration: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # 1. Determine "to_play"
        to_play = 0

        player_id = kwargs.get("player_id")
        if player_id is not None:
            if isinstance(player_id, int):
                to_play = player_id
            elif isinstance(player_id, str):
                try:
                    # Attempt to parse PettingZoo style "player_1" or just "1"
                    if "_" in player_id:
                        to_play = int(player_id.split("_")[-1])
                    else:
                        to_play = int(player_id)
                except ValueError:
                    pass

        # Fallback to info dict if not found
        if to_play == 0 and info:
            if "player" in info:
                to_play = info["player"]
            elif "to_play" in info:
                to_play = info["to_play"]

        # Get episode step for temperature
        # If exploration is explicitly False, we can skip temperature decay to avoid massive loops.
        episode_step = kwargs.get("episode_step", 0)
        curr_temp = (
            0.0 if exploration is False else self._get_current_temperature(episode_step)
        )

        # 3. Run MCTS
        # Search algorithms usually expect: run(state, info, to_play, agent_network)
        start_search_time = time.time()
        root_value, exploratory_policy, target_policy, best_action, search_metadata = (
            self.search.run(obs, info, to_play, agent_network)
        )
        search_duration = time.time() - start_search_time
        mcts_sps = (
            self.config.num_simulations / search_duration
            if search_duration > 0
            else 0.0
        )
        search_metadata["mcts_sps"] = mcts_sps

        from modules.world_models.inference_output import InferenceOutput

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
            **kwargs,
        )

        metadata = {
            # Convert to standard Python list so PyTorch can instantly free the tensor memory
            "policy": (
                target_policy.tolist()
                if isinstance(target_policy, torch.Tensor)
                else target_policy
            ),
            "value": float(root_value),
            "best_action": int(action),
            # DO NOT save the whole search_metadata dict!
            # Only extract the specific float you need for logging so the rest can be garbage collected.
            "search_metadata": {
                "mcts_sps": float(search_metadata.get("mcts_sps", 0.0))
            },
            "root_value": float(root_value),
        }

        return action, metadata

    def mask_actions(
        self,
        values: torch.Tensor,
        legal_moves: Any,
        mask_value: float = -float("inf"),
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return self.inner_selector.mask_actions(values, legal_moves, mask_value, device)

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        self.inner_selector.update_parameters(params_dict)
