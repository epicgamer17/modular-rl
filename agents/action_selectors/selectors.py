from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import torch
import numpy as np

# Constant for default epsilon
DEFAULT_EPSILON = 0.05


# TODO: remove default exploration from select_action, no more self.exploration on these.
class BaseActionSelector(ABC):
    @abstractmethod
    def select_action(
        self,
        agent_network: torch.nn.Module,
        obs: Any,
        info: Optional[Dict[str, Any]] = None,
        network_output: Optional[Any] = None,
        exploration: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError

    def mask_actions(
        self,
        values: torch.Tensor,
        legal_moves: Any,
        mask_value: float = -float("inf"),
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Masks illegal actions in the given values (logits, probs, or Q-values).

        Args:
            values: The tensor to mask [B, A] or [A].
            legal_moves: List of legal move indices or list of lists for batched input.
            mask_value: The value to use for masking (default -inf).
            device: Optional device.

        Returns:
            The masked tensor.
        """
        if device is None:
            device = values.device

        # Core masking logic (adapted from utils.action_mask)
        mask = torch.zeros_like(values, dtype=torch.bool).to(device)

        if values.dim() == 1:
            if isinstance(legal_moves, (list, np.ndarray, torch.Tensor)):
                mask[legal_moves] = True
            else:
                raise ValueError(
                    f"For 1D actions, legal_moves must be an iterable of indices, got {type(legal_moves)}"
                )
        elif values.dim() == 2:
            # Batch of legal moves: [[...], [...]]
            # Special case: if batch size is 1 and legal_moves is a single list of moves
            if (
                values.shape[0] == 1
                and len(legal_moves) > 0
                and not isinstance(legal_moves[0], (list, np.ndarray, torch.Tensor))
            ):
                mask[0, legal_moves] = True
            else:
                for i, legal in enumerate(legal_moves):
                    if legal is not None:
                        mask[i, legal] = True
        else:
            raise ValueError(
                f"mask_actions expects 1D or 2D tensor, got {values.dim()}D"
            )

        return torch.where(mask, values, torch.tensor(mask_value, device=device))

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        """
        Updates the internal parameters of the selector.

        Args:
            params_dict: Dictionary containing parameter updates.
        """
        pass


class CategoricalSelector(BaseActionSelector):
    def __init__(self, exploration: bool = True):
        # We keep this for backward compatibility with SelectorFactory/Configs that might pass it,
        # but select_action argument takes precedence.
        self.default_exploration = exploration

    def select_action(
        self,
        agent_network,
        obs,
        info: Optional[Dict[str, Any]] = None,
        network_output: Optional[Any] = None,
        exploration: Optional[bool] = None,
        **kwargs,
    ):
        if network_output is None:
            network_output = agent_network.obs_inference(obs)

        # Resolve exploration flag
        # If explicitly passed, use it. Otherwise use default from init.
        # If neither (e.g. None passed and config didn't specify), default to True?
        # Typically PPO config specifies 'exploration': True in kwargs.
        should_explore = (
            exploration if exploration is not None else self.default_exploration
        )

        metadata = {}
        # 1. Action Masking Logic
        policy = network_output.policy
        legal_moves = info.get("legal_moves") if info is not None else None

        if legal_moves is not None:
            from torch.distributions import Categorical

            # Apply mask to logits (-inf for illegal moves)
            # Categorical distribution usually has logits or probs.
            # We prefer masking logits for numerical stability.
            if hasattr(policy, "logits") and policy.logits is not None:
                logits = policy.logits
            else:
                # Fallback to probs if logits are not available
                logits = torch.log(policy.probs + 1e-8)

            masked_logits = self.mask_actions(
                logits,
                legal_moves,
                mask_value=-float("inf"),
                device=logits.device,
            )
            # 2. Repackage the Distribution
            policy = Categorical(logits=masked_logits)

        # Always store distribution for decorators (e.g. PPODecorator)
        metadata["policy"] = policy

        if should_explore:
            action = policy.sample()
        else:
            # Assumes policy has 'probs' attribute, usually true for Categorical distribution
            action = torch.argmax(policy.probs, dim=-1)

        return action, metadata


class EpsilonGreedySelector(BaseActionSelector):
    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def select_action(
        self,
        agent_network,
        obs,
        info: Optional[Dict[str, Any]] = None,
        network_output: Optional[Any] = None,
        exploration: Optional[bool] = None,
        **kwargs,
    ):
        if network_output is None:
            network_output = agent_network.obs_inference(obs)

        q_values = network_output.q_values
        batch_size = q_values.shape[0] if q_values.dim() == 2 else 1

        # Check if legal moves are provided
        legal_moves = None
        if info is not None:
            legal_moves = info.get("legal_moves")

        # Determine effective epsilon
        should_explore = exploration if exploration is not None else True
        effective_epsilon = self.epsilon if should_explore else 0.0

        # Epsilon-greedy logic with batched independent exploration
        random_vals = torch.rand(batch_size, device=q_values.device)
        explore_mask = random_vals < effective_epsilon

        # Greedy action (with masking if needed)
        if legal_moves is not None:
            masked_values = self.mask_actions(
                q_values,
                legal_moves,
                mask_value=-float("inf"),
                device=q_values.device,
            )
            greedy_actions = masked_values.argmax(dim=-1)
        else:
            greedy_actions = q_values.argmax(dim=-1)

        if not should_explore or effective_epsilon == 0:
            return greedy_actions, {}

        # Handle exploration
        if batch_size == 1:
            if explore_mask.item():
                if legal_moves is not None and len(legal_moves) > 0:
                    # legal_moves could be a single list [idx1, idx2] or [[idx1, idx2]]
                    actual_legal = (
                        legal_moves[0] if q_values.dim() == 2 else legal_moves
                    )
                    action = torch.tensor(
                        np.random.choice(actual_legal), device=q_values.device
                    )
                else:
                    action = torch.randint(
                        0, q_values.shape[-1], (), device=q_values.device
                    )
                return action, {}
            return greedy_actions, {}
        else:
            # Batched case
            actions = greedy_actions.clone()
            for i in range(batch_size):
                if explore_mask[i]:
                    if legal_moves is not None and legal_moves[i] is not None:
                        actions[i] = np.random.choice(legal_moves[i])
                    else:
                        actions[i] = torch.randint(
                            0, q_values.shape[-1], (), device=q_values.device
                        )
            return actions, {}

    def update_parameters(self, params: Dict[str, Any]) -> None:
        if "epsilon" in params:
            self.epsilon = float(params["epsilon"])


class ArgmaxSelector(BaseActionSelector):
    """
    Selects the action with the highest value/logit.
    Essentially EpsilonGreedy with epsilon=0.
    """

    def select_action(
        self,
        agent_network: torch.nn.Module,
        obs: Any,
        info: Optional[Dict[str, Any]] = None,
        network_output: Optional[Any] = None,
        exploration: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if network_output is None:
            network_output = agent_network.obs_inference(obs)

        # Check for legal moves
        legal_moves = None
        if info is not None:
            legal_moves = info.get("legal_moves")

        # Priority: q_values > policy probs > policy logits
        values = None
        if hasattr(network_output, "q_values") and network_output.q_values is not None:
            values = network_output.q_values
        elif hasattr(network_output, "policy") and network_output.policy is not None:
            if hasattr(network_output.policy, "probs"):
                values = network_output.policy.probs
            elif hasattr(network_output.policy, "logits"):
                values = network_output.policy.logits
        else:
            raise ValueError("No values found in network output")
        # DOES NOT HANDLE VALUE, VALUE IS A SCALAR FOR THE CURRENT STATE
        if legal_moves is not None and len(legal_moves) > 0:
            values = self.mask_actions(
                values,
                legal_moves,
                mask_value=-float("inf"),
                device=values.device,
            )

        action = values.argmax(dim=-1)
        return action, {}


class NFSPSelector(BaseActionSelector):
    """
    NFSPSelector manages the selection between Best Response (RL)
    and Average Strategy (SL) policies based on the anticipatory parameter (eta).
    """

    def __init__(
        self,
        br_selector: BaseActionSelector,
        avg_selector: BaseActionSelector,
        eta: float = 0.1,
    ):
        self.br_selector = br_selector
        self.avg_selector = avg_selector
        self.eta = eta

    def select_action(
        self,
        agent_network: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
        obs: Any,
        info: Optional[Dict[str, Any]] = None,
        network_output: Optional[Any] = None,
        exploration: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # Decide which policy to use
        # eta = P(Best Response)
        import random

        should_use_br = random.random() < self.eta

        # Handle agent_network as a dict or ModuleDict
        if isinstance(agent_network, (dict, torch.nn.ModuleDict)):
            br_net = agent_network["best_response"]
            avg_net = agent_network["average_strategy"]

            # If nested (separate networks per player)
            player_id = kwargs.get("player_id")
            if player_id is not None:
                if isinstance(br_net, (dict, torch.nn.ModuleDict)):
                    br_net = br_net[player_id]
                if isinstance(avg_net, (dict, torch.nn.ModuleDict)):
                    avg_net = avg_net[player_id]
        else:
            # Fallback for single network (not typical for NFSP but good for robust testing)
            br_net = agent_network
            avg_net = agent_network

        if should_use_br:
            action, metadata = self.br_selector.select_action(
                br_net, obs, info, exploration=exploration, **kwargs
            )
            metadata["policy_used"] = "best_response"
        else:
            action, metadata = self.avg_selector.select_action(
                avg_net, obs, info, exploration=exploration, **kwargs
            )
            metadata["policy_used"] = "average_strategy"

        return action, metadata

    def update_parameters(self, params: Dict[str, Any]) -> None:
        if "eta" in params:
            self.eta = float(params["eta"])
        self.br_selector.update_parameters(params)
        self.avg_selector.update_parameters(params)
