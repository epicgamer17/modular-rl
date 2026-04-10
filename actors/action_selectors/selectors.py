from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import torch
import numpy as np

# Constant for default epsilon
DEFAULT_EPSILON = 0.05


class BaseActionSelector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select_action(
        self,
        predictions: Dict[str, torch.Tensor],
        info: Dict[str, Any],
        exploration: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError  # pragma: no cover

    def mask_actions(
        self,
        values: torch.Tensor,
        legal_moves: Any,
        mask_value: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Masks illegal actions in the given values (logits, probs, or Q-values).

        Args:
            values: The tensor to mask [B, A] or [A].
            legal_moves: List of legal move indices or list of lists for batched input.
            mask_value: The value to use for masking (defaults to self.config.default_mask_value).
            device: Optional device.

        Returns:
            The masked tensor.
        """
        if mask_value is None:
            mask_value = -float("inf")

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
            # Special case: if batch size is 1 and legal_moves is a single list of moves OR a 1D tensor mask
            if values.shape[0] == 1 and len(legal_moves) > 0:
                # If it's a list, check if the first element is a list to determine nesting
                is_nested = isinstance(legal_moves[0], (list, np.ndarray)) or (
                    torch.is_tensor(legal_moves) and legal_moves.dim() > 1
                )

                if not is_nested:
                    mask[0, legal_moves] = True
                else:
                    mask[0, legal_moves[0]] = True
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
        pass  # pragma: no cover


class CategoricalSelector(BaseActionSelector):
    def __init__(self, exploration: bool = True):
        super().__init__()
        self.default_exploration = exploration

    def select_action(
        self,
        predictions: Dict[str, torch.Tensor],
        info: Dict[str, Any],
        exploration: Optional[bool] = None,
        **kwargs,
    ):
        # Resolve exploration flag
        should_explore = (
            exploration if exploration is not None else self.default_exploration
        )

        metadata = {}

        mask = info.get("legal_moves_mask", info.get("legal_moves"))

        from torch.distributions import Categorical

        logits = predictions.get("logits")
        probs = predictions.get("probs")

        if logits is not None:
            if mask is not None:
                logits = self.mask_actions(
                    logits, mask, mask_value=-float("inf"), device=logits.device
                )
            policy = Categorical(logits=logits)
        else:
            assert (
                probs is not None
            ), "CategoricalSelector requires logits or probs in predictions"
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
    def __init__(self, epsilon: float = 0.05):
        super().__init__()
        self.epsilon = epsilon

    def select_action(
        self,
        predictions: Dict[str, torch.Tensor],
        info: Dict[str, Any],
        exploration: Optional[bool] = None,
        **kwargs,
    ):
        q_values = predictions.get("q_values")
        assert (
            q_values is not None
        ), "EpsilonGreedySelector requires q_values in predictions"
        batch_size = q_values.shape[0] if q_values.dim() == 2 else 1

        # Check if legal moves are provided
        mask = info.get("legal_moves_mask", info.get("legal_moves"))
        if mask is not None:
            q_values = self.mask_actions(
                q_values, mask, mask_value=-float("inf"), device=q_values.device
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
                if not torch.is_tensor(mask):
                    # If it's a list of indices, create a binary mask
                    new_mask = torch.zeros(
                        (batch_size, q_values.shape[-1]), device=q_values.device
                    )

                    if q_values.dim() == 1:
                        new_mask[mask] = 1.0
                    else:
                        # Batch size > 1 or Batch size 1 with possible nesting
                        if batch_size == 1 and len(mask) > 0:
                            is_nested = isinstance(
                                mask[0], (list, np.ndarray, torch.Tensor)
                            )
                            if not is_nested:
                                new_mask[0, mask] = 1.0
                            else:
                                new_mask[0, mask[0]] = 1.0
                        else:
                            for i, m in enumerate(mask):
                                if m is not None:
                                    new_mask[i, m] = 1.0
                    mask = new_mask

                probs = mask.float()
                # Ensure there's at least one legal move to sample from
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
        predictions: Dict[str, torch.Tensor],
        info: Dict[str, Any],
        exploration: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        mask = info.get("legal_moves_mask", info.get("legal_moves"))

        # Prefer q_values, fall back to logits or probs
        values = predictions.get("q_values")
        if values is None:
            values = predictions.get("logits")
            if values is None:
                values = predictions.get("probs")
        
        assert (
            values is not None
        ), "ArgmaxSelector requires q_values, logits, or probs in predictions"

        if mask is not None:
            values = self.mask_actions(values, mask)

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
        predictions: Dict[str, torch.Tensor],
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
                predictions, info, exploration=exploration, **kwargs
            )
            metadata.update(inner_metadata)
            metadata["policy_used"] = "best_response"
        else:
            action, inner_metadata = self.avg_selector.select_action(
                predictions, info, exploration=exploration, **kwargs
            )
            metadata.update(inner_metadata)
            metadata["policy_used"] = "average_strategy"

        return action, metadata

    def update_parameters(self, params: Dict[str, Any]) -> None:
        if "eta" in params:
            self.eta = float(params["eta"])
        self.br_selector.update_parameters(params)
        self.avg_selector.update_parameters(params)
