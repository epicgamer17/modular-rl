import torch
import time
from typing import Any, Optional, Tuple, TYPE_CHECKING
from core import PipelineComponent, Blackboard
from actors.action_selectors.types import InferenceResult

if TYPE_CHECKING:
    from modules.agent_nets.base import BaseAgentNetwork


class NetworkInferenceComponent(PipelineComponent):
    """
    Performs neural network inference for an actor.
    Reads 'obs' from data, writes results to the specified output key in predictions.
    """

    def __init__(
        self,
        agent_network: "BaseAgentNetwork",
        input_shape: Tuple[int, ...],
        output_key: str = "inference_result",
    ):
        self.agent_network = agent_network
        self.input_shape = input_shape
        self.output_key = output_key

    def execute(self, blackboard: Blackboard) -> None:
        obs = blackboard.data["obs"]
        if obs is None:
            return

        # Ensure batch dimension [1, ...] if single observation
        if obs.dim() == len(self.input_shape):
            obs = obs.unsqueeze(0)

        with torch.inference_mode():
            output = self.agent_network.obs_inference(obs)
            result = InferenceResult.from_inference_output(output)

        blackboard.predictions[self.output_key] = result
        # Also set default keys for single-inference components
        if self.output_key == "inference_result":
            blackboard.predictions["logits"] = result.logits
            blackboard.predictions["value"] = result.value


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

                result = InferenceResult(
                    probs=probs,
                    value=values,
                    extra_metadata={
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
                    },
                )
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

                result = InferenceResult(
                    probs=probs,
                    value=value,
                    extra_metadata={
                        "target_policies": target_policies_out,
                        "search_duration": time.time() - start_time,
                        "search_metadata": search_metadata,
                        "best_actions": best_actions_out,
                        "value": value.squeeze(0),
                        "root_value": value.squeeze(0),
                    },
                )

        blackboard.predictions["logits"] = result.logits
        blackboard.predictions["value"] = result.value
        blackboard.predictions["inference_result"] = result
