from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
import torch
import time
import numpy as np

from agents.action_selectors.types import InferenceResult
from modules.agent_nets.base import BaseAgentNetwork
from modules.world_models.inference_output import InferenceOutput


class BasePolicySource(ABC):
    """
    Abstract base class for providing InferenceResults to ActionSelectors.
    Encapsulates the difference between raw network inference and search-based (MCTS) policy.
    """

    @abstractmethod
    def get_inference(
        self, obs: torch.Tensor, info: Dict[str, Any], **kwargs
    ) -> InferenceResult:
        """
        Computes or retrieves an InferenceResult for the given observation.
        """
        pass


class NetworkPolicySource(BasePolicySource):
    """
    Policy source that performs a pure forward pass on a neural network.
    """

    def __init__(self, agent_network: BaseAgentNetwork):
        self.agent_network = agent_network

    def get_inference(
        self, obs: torch.Tensor, info: Dict[str, Any], **kwargs
    ) -> InferenceResult:
        """
        Runs obs_inference on the network and converts the result to an InferenceResult.
        """
        output = self.agent_network.obs_inference(obs)
        return InferenceResult.from_inference_output(output)


class SearchPolicySource(BasePolicySource):
    """
    Policy source that wraps an MCTS search engine.
    Maps search visit counts (exploratory policy) to the InferenceResult contract.

    Accepts agent_network via kwargs at inference time so callers can pass
    the current (potentially updated) network without storing it here.
    """

    def __init__(
        self, search_engine: Any, agent_network: Optional[BaseAgentNetwork], config: Any = None
    ):
        self.search = search_engine
        self.agent_network = agent_network
        self.config = config

    def get_inference(
        self, obs: torch.Tensor, info: Dict[str, Any], **kwargs
    ) -> InferenceResult:
        """
        Runs MCTS and wraps results into an InferenceResult.

        Accepts agent_network via kwargs (takes precedence over self.agent_network).
        """
        agent_network = kwargs.pop("agent_network", self.agent_network)
        to_play = kwargs.get("to_play", 0)
        exploration = kwargs.get("exploration", True)

        # MCTSDecorator logic uses run_vectorized if B > 1
        is_batched = (
            obs.dim() > len(agent_network.input_shape) and obs.shape[0] > 1
        )

        start_time = time.time()

        if is_batched:
            infos_list = info.get("infos_list", [])
            if not infos_list and "legal_moves" in info:
                B = obs.shape[0]
                infos_list = [
                    {
                        "legal_moves": info["legal_moves"][i],
                        "player": info.get("player", [0] * B)[i],
                    }
                    for i in range(B)
                ]

            res = self.search.run_vectorized(
                obs, infos_list, to_play, agent_network
            )
            (
                root_values,
                exploratory_policies,
                target_policies,
                best_actions,
                sm_list,
            ) = res

            search_duration = time.time() - start_time

            probs = torch.stack(
                [
                    torch.as_tensor(p, device=obs.device, dtype=torch.float32)
                    for p in exploratory_policies
                ]
            )
            values = torch.as_tensor(
                root_values, device=obs.device, dtype=torch.float32
            )

            return InferenceResult(
                probs=probs,
                value=values,
                extra_metadata={
                    "target_policies": target_policies,
                    "search_duration": search_duration,
                    "search_metadata": sm_list,
                    "best_actions": best_actions,
                },
            )
        else:
            res = self.search.run(
                obs, info, to_play, agent_network, exploration=exploration
            )
            (
                root_value,
                exploratory_policy,
                target_policy,
                best_action,
                search_metadata,
            ) = res

            search_duration = time.time() - start_time

            return InferenceResult(
                probs=exploratory_policy.to(obs.device),
                value=torch.tensor(
                    [root_value], device=obs.device, dtype=torch.float32
                ),
                extra_metadata={
                    "target_policy": target_policy,
                    "search_duration": search_duration,
                    "search_metadata": search_metadata,
                    "best_action": best_action,
                },
            )
