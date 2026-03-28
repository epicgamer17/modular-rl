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
        **kwargs,
    ):
        self.search = search_engine
        self.agent_network = agent_network

    @torch.inference_mode()
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

        # Always shallow copy info to prevent in-place mutations leaking back to actor/env
        if isinstance(info, dict):
            info = info.copy()
        elif isinstance(info, list):
            info = [i.copy() if isinstance(i, dict) else i for i in info]

        # Populate info["player"] from to_play kwarg if not already present
        to_play = kwargs.get("to_play")
        if to_play is not None:
            if isinstance(info, dict):
                if "player" not in info:
                    info["player"] = to_play
            elif isinstance(info, list):
                # If it's a list, we might need to update each element
                new_info = []
                for i, item in enumerate(info):
                    if isinstance(item, dict) and "player" not in item:
                        # If to_play is a list matching info, use corresponding element
                        p = to_play[i] if (isinstance(to_play, (list, np.ndarray, torch.Tensor)) and len(to_play) == len(info)) else to_play
                        item["player"] = p
                    new_info.append(item)
                info = new_info

        assert (
            (isinstance(info, dict) and "player" in info) or
            (isinstance(info, list) and all(isinstance(i, dict) and "player" in i for i in info))
        ), "info must contain 'player' in all entries, or pass to_play as a kwarg"

        start_time = time.time()
        
        # Optimization: Use search.run for batch size 1 to avoid redundant list processing
        if obs.shape[0] == 1:
            single_info = info[0] if isinstance(info, list) else info
            # modular_search.run() expects (obs, info, agent_network)
            res_single = self.search.run(obs, single_info, agent_network, exploration=exploration)
            # res_single is (root_value, exploratory_policy, target_policy, best_action, search_metadata)
            
            search_duration = time.time() - start_time
            
            # Construct tensors with [1, ...] shapes - Detach for safety
            probs = torch.as_tensor(res_single[1], device=obs.device, dtype=torch.float32).detach().unsqueeze(0)
            value = torch.as_tensor([res_single[0]], device=obs.device, dtype=torch.float32).detach().unsqueeze(-1)
            action = torch.as_tensor([res_single[3]], device=obs.device, dtype=torch.long).detach()
            
            # Metadata keys for target policies and search metadata
            target_policies_tensor = torch.as_tensor(res_single[2], device=obs.device, dtype=torch.float32).detach().unsqueeze(0)
            
            return InferenceResult(
                probs=probs,
                value=value,
                action=action,
                extras={
                    "target_policies": target_policies_tensor,
                    "search_duration": search_duration,
                    "search_metadata": [res_single[4]],
                    "best_actions": action,
                    "value": value.squeeze(-1), # [1]
                    "root_value": value.squeeze(-1), # [1]
                },
            )

        # Vectorized path for B > 1
        res = self.search.run_vectorized(obs, info, agent_network, exploration=exploration)
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
                ).detach().contiguous()
                for p in exploratory_policies
            ]
        )
        values = torch.as_tensor(
            root_values, device=obs.device, dtype=torch.float32
        ).detach()

        # Standardize values to [B, 1] for potential Multi-Player Reward mapping
        if values.dim() == 1:
            values = values.unsqueeze(-1)

        # Standardize search results to tensors for BaseActor squeezing and PufferActor indexing
        target_policies_tensor = torch.stack(
            [
                torch.as_tensor(
                    p, device=obs.device, dtype=torch.float32
                ).detach().contiguous()
                for p in target_policies
            ]
        )
        best_actions_tensor = torch.as_tensor(
            best_actions, device=obs.device, dtype=torch.long
        ).detach()

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
