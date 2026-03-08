from abc import ABC, abstractmethod
from typing import Optional, Any
import numpy as np
import torch
from search.nodes import ChanceNode, DecisionNode
from search.scoring_methods import PriorScoring, ScoringMethod
import torch.nn.functional as F


class SelectionStrategy(ABC):
    @abstractmethod
    def select_child(self, node, min_max_stats, pruned_searchset=None):
        pass  # pragma: no cover

    @abstractmethod
    def mask_actions(
        self,
        values: torch.Tensor,
        legal_moves: Any,
        mask_value: float = -float("inf"),
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Masks illegal actions in the given values (logits or scores).
        """
        pass  # pragma: no cover


class TopScoreSelection(SelectionStrategy):
    """
    Selects the child with the highest score according to the scoring_method.
    Supports a tiebreak_scoring_method for cases where scores are identical.
    """

    def __init__(
        self,
        scoring_method: ScoringMethod,
        tiebreak_scoring_method: Optional[ScoringMethod] = None,
    ):
        self.scoring_method = scoring_method
        self.tiebreak_scoring_method = tiebreak_scoring_method

    def select_child(self, node, min_max_stats, pruned_searchset=None):
        # assert isinstance(node, DecisionNode)
        # Relaxed for parity: allow ChanceNodes as well
        assert hasattr(
            node, "is_decision"
        ), f"Node must have is_decision, got {type(node)}"
        # assert node.expanded(), "node must be expanded to select a child"

        scores = self.scoring_method.get_scores(node, min_max_stats)

        # Always mask strictly illegal actions (prior == 0)
        # unless node is a ChanceNode where all priors might be small but > 0
        if node.is_decision:
            illegal_mask = node.child_priors <= 0
            if illegal_mask.any():
                scores[illegal_mask] = -1e18  # Use a very large negative value

        # Masking for pruned_searchset
        if pruned_searchset is not None:
            scores = self.mask_actions(scores, pruned_searchset)

        # Find max score
        max_score = torch.max(scores)

        # Identify ties
        # using a small epsilon for float comparison stability if needed,
        # but pure equality often works for discrete logic like visits.
        # safe approach: isclose
        tied_mask = torch.isclose(scores, max_score)

        # If we have a tiebreaker
        if self.tiebreak_scoring_method is not None:
            # Check if we actually have ties (sum of mask > 1)
            if tied_mask.sum() > 1:
                sec_scores = self.tiebreak_scoring_method.get_scores(
                    node, min_max_stats
                )
                # apply mask to clear non-tied actions
                sec_scores[~tied_mask] = -float("inf")

                max_sec = torch.max(sec_scores)
                tied_mask = torch.isclose(sec_scores, max_sec)

        # Break ties randomly among survivors
        tied_indices = torch.nonzero(tied_mask).flatten()

        if len(tied_indices) == 0:
            # Should not happen unless all scores are -inf
            # Fallback to argmax
            action = torch.argmax(scores).item()
        elif len(tied_indices) == 1:
            action = tied_indices[0].item()
        else:
            idx = torch.randint(len(tied_indices), (1,)).item()
            action = tied_indices[idx].item()

        return action, node.get_child(action)

    def mask_actions(
        self,
        values: torch.Tensor,
        legal_moves: Any,
        mask_value: float = -float("inf"),
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = values.device

        mask = torch.full_like(values, mask_value, device=device)

        if values.dim() == 1:
            if isinstance(legal_moves, (list, np.ndarray, torch.Tensor)):
                mask[legal_moves] = 0
            else:
                # Assuming legal_moves is already a list of indices
                mask[legal_moves] = 0
        elif values.dim() == 2:
            for i, legal in enumerate(legal_moves):
                if legal is not None:
                    mask[i, legal] = 0
        else:
            raise ValueError(
                f"mask_actions expects 1D or 2D tensor, got {values.dim()}D"
            )

        return values + mask


class SamplingSelection(SelectionStrategy):
    """
    Selects a child by sampling.
    For DecisionNodes: Samples based on scores from scoring_method (softmax or direct if prob).
    For ChanceNodes: Samples codes based on probabilities.
    """

    def __init__(
        self, scoring_method: Optional[ScoringMethod] = None, temperature: float = 1.0
    ):
        # TODO BETTER WAY OF ENFORCING THE SCORING METHOD IS A PROBABILITY, FOR EXAMPLE SAMPLING FROM IMPROVED GUMBEL POLICY
        # assert isinstance(scoring_method, PriorScoring)
        self.scoring_method = scoring_method
        self.temperature = temperature

    def select_child(self, node, min_max_stats, pruned_searchset=None):
        if isinstance(node, DecisionNode):
            scores = self.scoring_method.get_scores(node, min_max_stats)

            # Always mask strictly illegal actions (prior == 0)
            illegal_mask = node.child_priors <= 0
            if illegal_mask.any():
                scores[illegal_mask] = -1e18  # Use a very large negative value

            if pruned_searchset is not None:
                scores = self.mask_actions(scores, pruned_searchset)

            if self.temperature == 0:
                action = torch.argmax(scores).item()
            else:
                probs = scores.clone()
                # FIX: Clamp to 0.0 to prevent masked actions (-1e18) from causing negative probabilities
                probs = probs.clamp(min=0.0)

                if self.temperature != 1.0:
                    # Apply temperature scaling
                    probs = torch.pow(probs.clamp(min=1e-8), 1.0 / self.temperature)

                # Safe Normalization
                sum_probs = probs.sum()
                probs = probs / sum_probs

                action = torch.multinomial(probs, 1).item()
            return action, node.get_child(action)

        elif isinstance(node, ChanceNode):
            # ChanceNode selection
            probs = node.child_priors
            # Normalize
            probs = probs / probs.sum()

            code = torch.multinomial(probs, 1).item()

            # Return scalar code now (modular_search will handle one-hot conversion if needed)
            return code, node.get_child(code)

    def mask_actions(
        self,
        values: torch.Tensor,
        legal_moves: Any,
        mask_value: float = -float("inf"),
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        # For scores, we use additive masking by default (assuming logits or similar)
        if device is None:
            device = values.device

        mask = torch.full_like(values, mask_value, device=device)

        if values.dim() == 1:
            mask[legal_moves] = 0
        elif values.dim() == 2:
            for i, legal in enumerate(legal_moves):
                if legal is not None:
                    mask[i, legal] = 0

        return values + mask
