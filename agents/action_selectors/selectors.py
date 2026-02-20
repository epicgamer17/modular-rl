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
        # Ensure obs is a tensor
        if not torch.is_tensor(obs):
            # Try to convert if it's a numpy array, but ideally caller handles this
            if isinstance(obs, np.ndarray):
                obs = torch.as_tensor(
                    obs,
                    device=next(agent_network.parameters()).device,
                    dtype=torch.float32,
                )
                if obs.dim() == len(agent_network.config.backbone.input_shape):
                    obs = obs.unsqueeze(0)
            else:
                raise ValueError(f"Observation must be a torch.Tensor, got {type(obs)}")
        """
        Selects an action based on the agent network and observation.

        Args:
            agent_network: The neural network used to compute values/logits.
            obs: The current observation.
            info: Optional dictionary containing additional information (e.g., legal moves).
            network_output: Optional pre-computed network output to avoid re-computation.
            exploration: Whether to use exploration policies (e.g. sampling, epsilon-greedy).
                         If False, should return deterministic/greedy action.
            **kwargs: Additional parameters for selection.

        Returns:
            A tuple containing:
            - action: The selected action tensor.
            - metadata: A dictionary containing metadata (e.g., log_probs, q_values).
        """
        pass

    @abstractmethod
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
        pass

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
            # Store masked distribution for decorators (e.g. PPODecorator)
            metadata["dist"] = policy

        if should_explore:
            action = policy.sample()
        else:
            # Assumes policy has 'probs' attribute, usually true for Categorical distribution
            action = torch.argmax(policy.probs, dim=-1)

        return action, metadata

    def mask_actions(
        self,
        values: torch.Tensor,
        legal_moves: Any,
        mask_value: float = -float("inf"),
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
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
            for i, legal in enumerate(legal_moves):
                if legal is not None:
                    mask[i, legal] = True
        else:
            raise ValueError(
                f"mask_actions expects 1D or 2D tensor, got {values.dim()}D"
            )

        return torch.where(mask, values, torch.tensor(mask_value, device=device))


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

        # Check if legal moves are provided
        legal_moves = None
        if info is not None:
            legal_moves = info.get("legal_moves")

        # Determine effective epsilon
        # Default exploration to True if None (EpsilonGreedy is usually exploratory)
        should_explore = exploration if exploration is not None else True
        effective_epsilon = self.epsilon if should_explore else 0.0

        # Epsilon-greedy logic
        if np.random.rand() < effective_epsilon:
            # Random action
            if legal_moves is not None and len(legal_moves) > 0:
                action = torch.tensor(np.random.choice(legal_moves), device=obs.device)
            else:
                # Assumes q_values is [B, A]
                action = torch.randint(
                    0, network_output.q_values.shape[-1], (), device=obs.device
                )
        else:
            # Greedy action
            # Apply masking if needed
            if legal_moves is not None and len(legal_moves) > 0:
                masked_values = self.mask_actions(
                    network_output.q_values,
                    legal_moves,
                    mask_value=-float("inf"),
                    device=obs.device,
                )
                action = masked_values.argmax(dim=-1)
            else:
                action = network_output.q_values.argmax(dim=-1)

        return action, {}

    def mask_actions(
        self,
        values: torch.Tensor,
        legal_moves: Any,
        mask_value: float = -float("inf"),
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = values.device

        mask = torch.zeros_like(values, dtype=torch.bool).to(device)

        if values.dim() == 1:
            if isinstance(legal_moves, (list, np.ndarray, torch.Tensor)):
                mask[legal_moves] = True
            else:
                raise ValueError(
                    f"For 1D actions, legal_moves must be an iterable of indices, got {type(legal_moves)}"
                )
        elif values.dim() == 2:
            for i, legal in enumerate(legal_moves):
                if legal is not None:
                    mask[i, legal] = True
        else:
            raise ValueError(
                f"mask_actions expects 1D or 2D tensor, got {values.dim()}D"
            )

        return torch.where(mask, values, torch.tensor(mask_value, device=device))

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

    def mask_actions(
        self,
        values: torch.Tensor,
        legal_moves: Any,
        mask_value: float = -float("inf"),
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        # Reuse common logic
        if device is None:
            device = values.device

        mask = torch.zeros_like(values, dtype=torch.bool).to(device)

        if values.dim() == 1:
            mask[legal_moves] = True
        elif values.dim() == 2:
            for i, legal in enumerate(legal_moves):
                if legal is not None:
                    mask[i, legal] = True

        return torch.where(mask, values, torch.tensor(mask_value, device=device))
