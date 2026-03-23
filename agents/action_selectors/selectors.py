from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import torch
import numpy as np
from modules.models.inference_output import InferenceOutput
from agents.action_selectors.types import InferenceResult

# Constant for default epsilon
DEFAULT_EPSILON = 0.05


class BaseActionSelector(ABC):
    def __init__(self, config: Optional[Any] = None):
        self.config = config

    @abstractmethod
    def select_action(
        self,
        result: InferenceResult,
        info: Dict[str, Any],
        exploration: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError  # pragma: no cover

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        """
        Updates the internal parameters of the selector.

        Args:
            params_dict: Dictionary containing parameter updates.
        """
        pass  # pragma: no cover


class CategoricalSelector(BaseActionSelector):
    def __init__(self, config: Optional[Any] = None, exploration: bool = True):
        super().__init__(config)
        # We keep this for backward compatibility with SelectorFactory/Configs that might pass it,
        # but select_action argument takes precedence.
        self.default_exploration = exploration

    def select_action(
        self,
        result: InferenceResult,
        info: Dict[str, Any],
        exploration: Optional[bool] = None,
        **kwargs,
    ):
        # Resolve exploration flag
        should_explore = (
            exploration if exploration is not None else self.default_exploration
        )

        metadata = {}

        mask = info.get("legal_moves_mask")

        from torch.distributions import Categorical

        if result.logits is not None:
            logits = result.logits
            if mask is not None:
                logits = torch.where(
                    mask, 
                    logits, 
                    torch.tensor(-float("inf"), device=logits.device, dtype=logits.dtype)
                )
            policy = Categorical(logits=logits)
        else:
            assert (
                result.probs is not None
            ), "CategoricalSelector requires result.logits or result.probs"
            probs = result.probs
            if mask is not None:
                probs = probs * mask.float()
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            policy = Categorical(probs=probs)

        # Use 'policy_dist' for decorators (like PPODecorator)
        # Use 'policy' for the replay buffer (must be a tensor/numpy)
        metadata["policy_dist"] = policy
        metadata["policy"] = policy.probs.detach()

        if should_explore:
            action = policy.sample()
        else:
            # policy.logits is the canonical form for argmax (log-scale preserves ordering)
            action = torch.argmax(policy.logits, dim=-1)

        return action, metadata


class EpsilonGreedySelector(BaseActionSelector):
    def __init__(self, config: Optional[Any] = None, epsilon: float = 0.05):
        super().__init__(config)
        self.epsilon = epsilon

    def select_action(
        self,
        result: InferenceResult,
        info: Dict[str, Any],
        exploration: Optional[bool] = None,
        **kwargs,
    ):
        assert (
            result.q_values is not None
        ), "EpsilonGreedySelector requires result.q_values"
        q_values = result.q_values
        batch_size = q_values.shape[0] if q_values.dim() == 2 else 1

        # Check if legal moves are provided
        mask = info.get("legal_moves_mask")
        if mask is not None:
            q_values = torch.where(
                mask, 
                q_values, 
                torch.tensor(-float("inf"), device=q_values.device, dtype=q_values.dtype)
            )

        # Exploration/Exploitation logic
        # Determine if exploration should happen based on 'exploration' arg or default epsilon
        should_explore = exploration if exploration is not None else (self.epsilon > 0)
        effective_epsilon = (
            kwargs.get("epsilon", self.epsilon) if should_explore else 0.0
        )

        if effective_epsilon > 0:
            # Batched epsilon greedy
            # Generate random actions
            if mask is not None:
                # Sample from legal actions using multinomial if mask is provided
                # Convert mask to float for multinomial (0 for illegal, 1 for legal)
                probs = mask.float()
                # Ensure there's at least one legal move to sample from
                # If a row in probs is all zeros, multinomial will fail.
                # We can add a small epsilon to all probs to avoid this, or handle it.
                # For now, assume valid masks where at least one action is legal.
                random_actions = torch.multinomial(probs, 1).squeeze(-1)
            else:
                # If no mask, sample uniformly from all actions
                random_actions = torch.randint(
                    0, q_values.shape[-1], (batch_size,), device=q_values.device
                )

            greedy_actions = torch.argmax(q_values, dim=-1)

            # Draw epsilon flags for each item in the batch
            r = torch.rand(batch_size, device=q_values.device)
            explore_mask = r < effective_epsilon

            actions = torch.where(explore_mask, random_actions, greedy_actions)
        else:
            # Pure exploitation (epsilon = 0)
            actions = torch.argmax(q_values, dim=-1)

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
        result: InferenceResult,
        info: Dict[str, Any],
        exploration: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        mask = info.get("legal_moves_mask")

        # Prefer q_values, fall back to logits or probs
        values = result.q_values
        if values is None:
            values = result.logits if result.logits is not None else result.probs
        assert (
            values is not None
        ), "ArgmaxSelector requires result.q_values, result.logits, or result.probs"

        if mask is not None:
            values = torch.where(
                mask, 
                values, 
                torch.tensor(-float("inf"), device=values.device, dtype=values.dtype)
            )

        action = torch.argmax(values, dim=-1)
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
        result: InferenceResult,
        info: Dict[str, Any],
        exploration: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # Decide which policy to use
        # eta = P(Best Response)
        import random

        should_use_br = random.random() < self.eta

        metadata = {}

        if should_use_br:
            action, inner_metadata = self.br_selector.select_action(
                result, info, exploration=exploration, **kwargs
            )
            metadata.update(inner_metadata)
            metadata["policy_used"] = "best_response"
        else:
            action, inner_metadata = self.avg_selector.select_action(
                result, info, exploration=exploration, **kwargs
            )
            metadata.update(inner_metadata)
            metadata["policy_used"] = "average_strategy"

        return action, metadata

    def update_parameters(self, params: Dict[str, Any]) -> None:
        if "eta" in params:
            self.eta = float(params["eta"])
        self.br_selector.update_parameters(params)
        self.avg_selector.update_parameters(params)
