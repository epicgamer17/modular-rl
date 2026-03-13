import time
from typing import Any, Dict, Optional, Tuple
import torch
from agents.action_selectors.selectors import BaseActionSelector
from torch.distributions import Categorical
from utils.schedule import create_schedule, Schedule
from modules.world_models.inference_output import InferenceOutput
from search import ModularSearch


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
        network_output: Optional[InferenceOutput] = None,
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
        dist = metadata.get("policy", network_output.policy)
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
        self,
        inner_selector: BaseActionSelector,
        search_algorithm: ModularSearch,
        config: Any,
    ):
        self.inner_selector = inner_selector
        self.search = search_algorithm
        self.config = config
        self.temperature_schedule = create_schedule(config.temperature_schedule)
        # Cache for temperature schedule to avoid O(N^2)
        self._last_step = -1

    def _get_current_temperature(self, steps_in_episode: int) -> float:
        """
        Determines the current temperature for MCTS policy based on the episode step.
        """
        if steps_in_episode > self._last_step:
            self.temperature_schedule.step(steps_in_episode - self._last_step)
            self._last_step = steps_in_episode
        elif steps_in_episode < self._last_step:
            # Fallback: recreate if we go backwards (e.g. new episode)
            self.temperature_schedule = create_schedule(
                self.config.temperature_schedule
            )
            self.temperature_schedule.step(steps_in_episode)
            self._last_step = steps_in_episode
        return self.temperature_schedule.get_value()

    def _get_current_training_temperature(self) -> float:
        """Determines exploration temperature based on training steps."""
        return self.temperature_schedule.get_value(self.config.training_steps)

    def select_action(
        self,
        agent_network: torch.nn.Module,
        obs: Any,
        info: Optional[Dict[str, Any]] = None,
        network_output: Optional[InferenceOutput] = None,
        exploration: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # 1. Determine "to_play"
        to_play = 0

        player_id = kwargs.get("player_id")
        if player_id is not None:
            if isinstance(player_id, int):
                to_play = player_id
            elif isinstance(player_id, (list, tuple)):
                to_play = list(player_id)
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
        if (to_play == 0 or to_play == [0]) and info:
            if "player" in info:
                to_play = info["player"]
            elif "to_play" in info:
                to_play = info["to_play"]

        # 1.5 Robust to_play handling for lists of strings (PettingZoo/Puffer)
        if isinstance(to_play, list):
            new_to_play = []
            for p in to_play:
                if isinstance(p, str):
                    try:
                        if "_" in p:
                            new_to_play.append(int(p.split("_")[-1]))
                        else:
                            new_to_play.append(int(p))
                    except ValueError:
                        new_to_play.append(0)
                else:
                    new_to_play.append(p)
            to_play = new_to_play

        # Get episode step for temperature
        episode_step = kwargs.get("episode_step", 0)
        curr_temp = (
            0.0 if exploration is False else self._get_current_temperature(episode_step)
        )

        from modules.world_models.inference_output import InferenceOutput

        # 2. Check for Batching
        # We only use the vectorized path if we have multiple environments (B > 1)
        # Some actors (like PettingZooActor) unsqueeze single observations, which can
        # trigger this check incorrectly if we don't check for B > 1.
        is_batched = obs.dim() > len(agent_network.input_shape) and obs.shape[0] > 1

        if is_batched:
            # Vectorized Search Path
            B = obs.shape[0]
            infos_list = [
                {"legal_moves": lm, "player": p}
                for lm, p in zip(info["legal_moves"], info["player"])
            ]
            start_search_time = time.time()

            self.search.config.policy_temperature = curr_temp

            res = self.search.run_vectorized(obs, infos_list, to_play, agent_network)
            search_duration = time.time() - start_search_time
            (
                root_values,
                exploratory_policies,
                target_policies,
                best_actions,
                sm_list,
            ) = res

            exploratory_policies_t = torch.stack(
                [
                    (
                        ep.to(obs.device)
                        if isinstance(ep, torch.Tensor)
                        else torch.as_tensor(ep, device=obs.device)
                    )
                    for ep in exploratory_policies
                ]
            )
            target_policies_t = torch.stack(
                [
                    (
                        tp.to(obs.device)
                        if isinstance(tp, torch.Tensor)
                        else torch.as_tensor(tp, device=obs.device)
                    )
                    for tp in target_policies
                ]
            )

            # Stack outputs for vectorized temperature and selection
            root_values_t = torch.as_tensor(
                root_values, device=obs.device, dtype=torch.float32
            )
            if curr_temp == 0.0:
                # Greedy Vectorized
                heated_probs = torch.zeros_like(exploratory_policies_t)
                batched_best_actions = torch.tensor(best_actions, device=obs.device)
                heated_probs.scatter_(1, batched_best_actions.unsqueeze(1), 1.0)
                mcts_policy_dist = Categorical(probs=heated_probs)
            elif curr_temp != 1.0:
                # Heated/Cooled Vectorized
                log_probs = torch.log(exploratory_policies_t + 1e-8)
                log_probs = log_probs / curr_temp
                heated_probs = torch.softmax(log_probs, dim=-1)
                mcts_policy_dist = Categorical(probs=heated_probs)
            else:
                mcts_policy_dist = Categorical(probs=exploratory_policies_t)

            mcts_output = InferenceOutput(value=root_values_t, policy=mcts_policy_dist)

            kwargs["temperature"] = curr_temp
            action, _ = self.inner_selector.select_action(
                agent_network,
                obs,
                info,
                network_output=mcts_output,
                exploration=exploration,
                **kwargs,
            )

            # Collate metadata
            metadata = {
                "policy": [
                    tp.tolist() if isinstance(tp, torch.Tensor) else tp
                    for tp in target_policies
                ],
                "value": [float(rv) for rv in root_values],
                "best_action": [int(a) for a in best_actions],
                "search_metadata": {
                    "mcts_simulations": self.config.num_simulations * B,
                    "mcts_search_time": search_duration,
                },
                "root_value": [float(rv) for rv in root_values],
            }

            return action, metadata

        # 3. Single Action Search Path
        start_search_time = time.time()

        self.search.config.policy_temperature = curr_temp

        root_value, exploratory_policy, target_policy, best_action, search_metadata = (
            self.search.run(obs, info, to_play, agent_network, exploration=exploration)
        )
        search_duration = time.time() - start_search_time
        # Apply temperature to probabilities
        if curr_temp == 0.0:
            # Greedy
            heated_probs = torch.zeros_like(exploratory_policy)
            heated_probs[best_action] = 1.0
            mcts_policy_dist = Categorical(probs=heated_probs)
        elif curr_temp != 1.0:
            # Heated/Cooled
            log_probs = torch.log(exploratory_policy + 1e-8)
            log_probs = log_probs / curr_temp
            heated_probs = torch.softmax(log_probs, dim=-1)
            mcts_policy_dist = Categorical(probs=heated_probs)
        else:
            # Standard
            mcts_policy_dist = Categorical(probs=exploratory_policy)

        mcts_output = InferenceOutput(
            value=torch.tensor(root_value), policy=mcts_policy_dist
        )

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
            "policy": (
                target_policy.tolist()
                if isinstance(target_policy, torch.Tensor)
                else target_policy
            ),
            "value": float(root_value),
            "best_action": int(action),
            "search_metadata": {
                "mcts_simulations": int(search_metadata.get("mcts_simulations", 0)),
                "mcts_search_time": float(search_metadata.get("mcts_search_time", 0.0)),
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
