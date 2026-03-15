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
        self,
        search_engine: Any,
        agent_network: Optional[BaseAgentNetwork],
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
        """
        exploration = kwargs.get("exploration", True)

        # Ensure player context and others are in info for the search engine
        assert "player" in info
        # MCTSDecorator logic uses run_vectorized if B > 1
        is_batched = (
            obs.dim() > len(self.agent_network.input_shape) and obs.shape[0] > 1
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

            res = self.search.run_vectorized(obs, infos_list, self.agent_network)
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

            # Ensure values is [B, 1] if input was batched
            if not isinstance(values, torch.Tensor):
                values = torch.tensor(values, device=obs.device, dtype=torch.float32)
            if values.dim() == 1:
                values = values.unsqueeze(-1)

            # Standardize search results to tensors for BaseActor squeezing and PufferActor indexing
            target_policies_tensor = torch.stack(
                [
                    torch.as_tensor(p, device=obs.device, dtype=torch.float32)
                    for p in target_policies
                ]
            )
            best_actions_tensor = torch.as_tensor(
                best_actions, device=obs.device, dtype=torch.long
            )

            return InferenceResult(
                probs=probs,
                value=values,
                extra_metadata={
                    "target_policies": target_policies_tensor,
                    "search_duration": search_duration,
                    "search_metadata": sm_list,
                    "best_actions": best_actions_tensor,
                    "value": values.squeeze(-1),  # Tensor for consistency
                    "root_value": values.squeeze(-1),
                },
            )
        else:
            res = self.search.run(
                obs, info, self.agent_network, exploration=exploration
            )
            (
                root_value,
                exploratory_policy,
                target_policy,
                best_action,
                search_metadata,
            ) = res

            search_duration = time.time() - start_time
            probs = exploratory_policy.to(obs.device)
            value = torch.tensor([root_value], device=obs.device, dtype=torch.float32)

            # If the input was unsqueezed [1, ...], the output should be [1, A]
            if obs.dim() > len(self.agent_network.input_shape):
                probs = probs.unsqueeze(0)
                # value is already [1] which matches [B]
                target_policies_out = target_policy.unsqueeze(0).to(obs.device)
                best_actions_out = torch.tensor([best_action], device=obs.device)
            else:
                target_policies_out = target_policy.to(obs.device)
                best_actions_out = torch.tensor(best_action, device=obs.device)

            return InferenceResult(
                probs=probs,
                value=value,
                extra_metadata={
                    "target_policies": target_policies_out,
                    "search_duration": search_duration,
                    "search_metadata": search_metadata,
                    "best_actions": best_actions_out,
                    "value": value.squeeze(0),
                    "root_value": value.squeeze(0),
                },
            )
