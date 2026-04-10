from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List, Tuple
import torch
import time
import numpy as np

from modules.agent_nets.base import BaseAgentNetwork
from modules.world_models.inference_output import InferenceOutput


class BasePolicySource(ABC):
    """
    Abstract base class for providing inference predictions to ActionSelectors.
    Encapsulates the difference between raw network inference and search-based (MCTS) policy.
    """

    @abstractmethod
    def get_inference(
        self, obs: torch.Tensor, info: Dict[str, Any], **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Computes or retrieves predictions for the given observation.
        """
        pass


class NetworkPolicySource(BasePolicySource):
    """
    Policy source that performs a pure forward pass on a neural network.
    """

    def __init__(self, agent_network: BaseAgentNetwork, input_shape: Tuple[int, ...]):
        self.agent_network = agent_network
        self.input_shape = input_shape

    def get_inference(
        self, obs: torch.Tensor, info: Dict[str, Any], **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Runs obs_inference on the network and returns the result as a dictionary.
        """
        # Ensure batch dimension
        if obs.dim() == len(self.input_shape):
            obs = obs.unsqueeze(0)

        output = self.agent_network.obs_inference(obs)
        
        # Convert InferenceOutput to dict
        preds = {}
        
        q_values = getattr(output, "q_values", None)
        if q_values is not None:
            preds["q_values"] = q_values
            
        policy = getattr(output, "policy", None)
        if policy is not None:
            logits = getattr(policy, "logits", None)
            if logits is not None:
                preds["logits"] = logits
            else:
                probs = getattr(policy, "probs", None)
                if probs is not None:
                    preds["probs"] = probs

        value = getattr(output, "value", None)
        if value is not None:
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor([value], device=obs.device)
            preds["value"] = value

        reward = getattr(output, "reward", None)
        if reward is not None:
            if not isinstance(reward, torch.Tensor):
                reward = torch.as_tensor([reward], device=obs.device)
            preds["reward"] = reward

        to_play = getattr(output, "to_play", None)
        if to_play is not None:
            if not isinstance(to_play, torch.Tensor):
                to_play = torch.as_tensor([to_play], device=obs.device, dtype=torch.long)
            preds["to_play"] = to_play

        extras = getattr(output, "extras", None) or {}
        if extras:
            preds["extra_metadata"] = extras

        return preds


class SearchPolicySource(BasePolicySource):
    """
    Policy source that wraps an MCTS search engine.
    Maps search visit counts (exploratory policy) to the predictions contract.

    Accepts agent_network via kwargs at inference time so callers can pass
    the current (potentially updated) network without storing it here.
    """

    def __init__(
        self,
        search_engine: Any,
        agent_network: Optional[BaseAgentNetwork],
        input_shape: Tuple[int, ...],
        num_actions: int,
    ):
        self.search = search_engine
        self.agent_network = agent_network
        self.input_shape = input_shape
        self.num_actions = num_actions

    def get_inference(
        self, obs: torch.Tensor, info: Dict[str, Any], **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Runs MCTS and returns results in a dictionary.

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

        # MCTSDecorator logic uses run_vectorized if B > 1
        is_batched = obs.dim() > len(self.input_shape) and obs.shape[0] > 1

        start_time = time.time()

        if is_batched:
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

            return {
                "probs": probs,
                "value": values,
                "extra_metadata": {
                    "target_policies": target_policies_tensor,
                    "search_duration": search_duration,
                    "search_metadata": sm_list,
                    "best_actions": best_actions_tensor,
                    "value": values.squeeze(-1),  # Tensor for consistency
                    "root_value": values.squeeze(-1),
                },
            }
        else:
            res = self.search.run(obs, info, agent_network, exploration=exploration)
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
            if obs.dim() > len(self.input_shape):
                probs = probs.unsqueeze(0)
                # value is already [1] which matches [B]
                target_policies_out = target_policy.unsqueeze(0).to(obs.device)
                best_actions_out = torch.tensor([best_action], device=obs.device)
            else:
                target_policies_out = target_policy.to(obs.device)
                best_actions_out = torch.tensor(best_action, device=obs.device)

            return {
                "probs": probs,
                "value": value,
                "extra_metadata": {
                    "target_policies": target_policies_out,
                    "search_duration": search_duration,
                    "search_metadata": search_metadata,
                    "best_actions": best_actions_out,
                    "value": value.squeeze(0),
                    "root_value": value.squeeze(0),
                },
            }


class NFSPNetworkPolicySource(BasePolicySource):
    """Combines two networks into one predictions dictionary for NFSP.

    Returns BR information in `q_values` and AVG information in `logits`/`probs`,
    so an `NFSPSelector` can choose between `EpsilonGreedySelector` and
    `CategoricalSelector` without additional inference passes.
    """

    def __init__(
        self,
        best_response_network: BaseAgentNetwork,
        average_network: BaseAgentNetwork,
    ):
        self.best_response_network = best_response_network
        self.average_network = average_network

    def get_inference(
        self, obs: torch.Tensor, info: Dict[str, Any], **kwargs
    ) -> Dict[str, torch.Tensor]:
        br_preds = NetworkPolicySource(self.best_response_network, (0,)).get_inference(obs, info)
        avg_preds = NetworkPolicySource(self.average_network, (0,)).get_inference(obs, info)

        preds = {
            "value": br_preds.get("value") if br_preds.get("value") is not None else avg_preds.get("value"),
            "q_values": br_preds.get("q_values"),
            "logits": avg_preds.get("logits"),
            "probs": avg_preds.get("probs"),
            "reward": (
                br_preds.get("reward") if br_preds.get("reward") is not None else avg_preds.get("reward")
            ),
            "to_play": (
                br_preds.get("to_play")
                if br_preds.get("to_play") is not None
                else avg_preds.get("to_play")
            ),
        }
        
        # Merge metadata
        extra_metadata = {
            **(br_preds.get("extra_metadata", {})),
            **(avg_preds.get("extra_metadata", {})),
        }
        if extra_metadata:
            preds["extra_metadata"] = extra_metadata

        return preds
