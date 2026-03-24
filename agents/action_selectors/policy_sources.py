from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
import torch
import time
import numpy as np
from torch.distributions import Categorical

from agents.action_selectors.types import InferenceResult
from modules.models.agent_network import AgentNetwork
from modules.models.inference_output import InferenceOutput


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

    def __init__(self, agent_network: AgentNetwork):
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
        self,
        search_engine: Any,
        agent_network: Optional[AgentNetwork],
        config: Any = None,
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
        Accepts to_play via kwargs to populate info["player"] if not already set.
        """
        exploration = kwargs.get("exploration", True)
        agent_network = kwargs.get("agent_network", self.agent_network)

        # Populate info["player"] from to_play kwarg if not already present
        if "player" not in info and "to_play" in kwargs:
            info = dict(info)
            info["player"] = kwargs["to_play"]

        assert (
            "player" in info
        ), "info must contain 'player', or pass to_play as a kwarg"

        start_time = time.time()
        res = self.search.run_vectorized(obs, info, agent_network)
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
                torch.as_tensor(
                    p, device=obs.device, dtype=torch.float32
                ).contiguous()
                for p in exploratory_policies
            ]
        )
        values = torch.as_tensor(
            root_values, device=obs.device, dtype=torch.float32
        )

        # Standardize values to [B, 1] for potential Multi-Player Reward mapping
        if values.dim() == 1:
            values = values.unsqueeze(-1)

        # Standardize search results to tensors for BaseActor squeezing and PufferActor indexing
        target_policies_tensor = torch.stack(
            [
                torch.as_tensor(
                    p, device=obs.device, dtype=torch.float32
                ).contiguous()
                for p in target_policies
            ]
        )
        best_actions_tensor = torch.as_tensor(
            best_actions, device=obs.device, dtype=torch.long
        )

        return InferenceResult(
            probs=probs,
            value=values,
            action=best_actions_tensor,
            extras={
                "target_policies": target_policies_tensor,
                "search_duration": search_duration,
                "search_metadata": sm_list,
                "best_actions": best_actions_tensor,
                "value": values.squeeze(-1),
                "root_value": values.squeeze(-1),
            },
        )


class NFSPNetworkPolicySource(BasePolicySource):
    """Combines two networks into one InferenceResult for NFSP.

    Returns BR information in `q_values` and AVG information in `logits`/`probs`,
    so an `NFSPSelector` can choose between `EpsilonGreedySelector` and
    `CategoricalSelector` without additional inference passes.
    """

    def __init__(
        self,
        best_response_network: AgentNetwork,
        average_network: AgentNetwork,
    ):
        self.best_response_network = best_response_network
        self.average_network = average_network

    def get_inference(
        self, obs: torch.Tensor, info: Dict[str, Any], **kwargs
    ) -> InferenceResult:
        br_output = self.best_response_network.obs_inference(obs)
        avg_output = self.average_network.obs_inference(obs)

        br_result = InferenceResult.from_inference_output(br_output)
        avg_result = InferenceResult.from_inference_output(avg_output)

        return InferenceResult(
            value=br_result.value if br_result.value is not None else avg_result.value,
            q_values=br_result.q_values,
            logits=avg_result.logits,
            probs=avg_result.probs,
            extras={
                **(br_result.extras or {}),
                **(avg_result.extras or {}),
            },
        )
