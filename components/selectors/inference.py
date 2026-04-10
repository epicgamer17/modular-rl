import torch
import time
from typing import Any, Optional, Tuple, TYPE_CHECKING
from core import PipelineComponent, Blackboard

if TYPE_CHECKING:
    from modules.agent_nets.base import BaseAgentNetwork


class NetworkInferenceComponent(PipelineComponent):
    """
    Performs neural network inference for an actor.
    Reads 'obs' from data, writes results to predictions.
    """

    def __init__(
        self,
        agent_network: "BaseAgentNetwork",
        input_shape: Tuple[int, ...],
    ):
        self.agent_network = agent_network
        self.input_shape = input_shape

    def execute(self, blackboard: Blackboard) -> None:
        obs = blackboard.data["obs"]
        done = blackboard.data.get("done", False)
        if obs is None or done:
            return

        # Ensure batch dimension [1, ...] if single observation
        if obs.dim() == len(self.input_shape):
            obs = obs.unsqueeze(0)

        with torch.inference_mode():
            output = self.agent_network.obs_inference(obs)
            
            # Write results directly to blackboard
            q_values = getattr(output, "q_values", None)
            if q_values is not None:
                blackboard.predictions["q_values"] = q_values
                
            policy = getattr(output, "policy", None)
            if policy is not None:
                logits = getattr(policy, "logits", None)
                if logits is not None:
                    blackboard.predictions["logits"] = logits
                else:
                    probs = getattr(policy, "probs", None)
                    if probs is not None:
                        blackboard.predictions["probs"] = probs

            value = getattr(output, "value", None)
            if value is not None:
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor([value], device=obs.device)
                blackboard.predictions["value"] = value

            reward = getattr(output, "reward", None)
            if reward is not None:
                if not isinstance(reward, torch.Tensor):
                    reward = torch.as_tensor([reward], device=obs.device)
                blackboard.predictions["reward"] = reward

            to_play = getattr(output, "to_play", None)
            if to_play is not None:
                if not isinstance(to_play, torch.Tensor):
                    to_play = torch.as_tensor([to_play], device=obs.device, dtype=torch.long)
                blackboard.predictions["to_play"] = to_play

            extras = getattr(output, "extras", None) or {}
            if extras:
                blackboard.predictions["extra_metadata"] = extras


class SearchInferenceComponent(PipelineComponent):
    """
    Performs MCTS-based search inference for an actor.
    """

    def __init__(
        self,
        search_engine: Any,
        agent_network: Optional["BaseAgentNetwork"],
        input_shape: Tuple[int, ...],
        num_actions: int,
        exploration: bool = True,
    ):
        self.search = search_engine
        self.agent_network = agent_network
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.exploration = exploration

    def execute(self, blackboard: Blackboard) -> None:
        obs = blackboard.data["obs"]
        done = blackboard.data.get("done", False)
        if obs is None or done:
            return
        info = blackboard.data.get("info", {})

        # Determine player_id/to_play
        # PettingZooObservationComponent provides 'player_id'
        player_id = blackboard.data.get("player_id", 0)
        if "player" not in info:
            info = {**info, "player": player_id}

        start_time = time.time()

        # Handle batched vs non-batched search
        is_batched = obs.dim() > len(self.input_shape) and obs.shape[0] > 1

        with torch.inference_mode():
            if is_batched:
                res = self.search.run_vectorized(obs, info, self.agent_network)
                (
                    root_values,
                    exploratory_policies,
                    target_policies,
                    best_actions,
                    sm_list,
                ) = res

                probs = torch.stack(
                    [
                        torch.as_tensor(p, device=obs.device, dtype=torch.float32)
                        for p in exploratory_policies
                    ]
                )
                values = torch.as_tensor(
                    root_values, device=obs.device, dtype=torch.float32
                )
                if values.dim() == 1:
                    values = values.unsqueeze(-1)

                blackboard.predictions["probs"] = probs
                blackboard.predictions["value"] = values
                blackboard.predictions["extra_metadata"] = {
                    "target_policies": torch.stack(
                        [
                            torch.as_tensor(
                                p, device=obs.device, dtype=torch.float32
                            )
                            for p in target_policies
                        ]
                    ),
                    "search_duration": time.time() - start_time,
                    "search_metadata": sm_list,
                    "best_actions": torch.as_tensor(
                        best_actions, device=obs.device, dtype=torch.long
                    ),
                    "value": values.squeeze(-1),
                    "root_value": values.squeeze(-1),
                }
            else:
                res = self.search.run(
                    obs, info, self.agent_network, exploration=self.exploration
                )
                (
                    root_value,
                    exploratory_policy,
                    target_policy,
                    best_action,
                    search_metadata,
                ) = res

                probs = exploratory_policy.to(obs.device)
                value = torch.tensor(
                    [root_value], device=obs.device, dtype=torch.float32
                )

                if obs.dim() > len(self.input_shape):
                    probs = probs.unsqueeze(0)
                    target_policies_out = target_policy.unsqueeze(0).to(obs.device)
                    best_actions_out = torch.tensor([best_action], device=obs.device)
                else:
                    target_policies_out = target_policy.to(obs.device)
                    best_actions_out = torch.tensor(best_action, device=obs.device)

                blackboard.predictions["probs"] = probs
                blackboard.predictions["value"] = value
                blackboard.predictions["extra_metadata"] = {
                    "target_policies": target_policies_out,
                    "search_duration": time.time() - start_time,
                    "search_metadata": search_metadata,
                    "best_actions": best_actions_out,
                    "value": value.squeeze(0),
                    "root_value": value.squeeze(0),
                }
