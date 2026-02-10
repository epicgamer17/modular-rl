from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import torch
from search.nodes import ChanceNode, DecisionNode
from search.scoring_methods import PriorScoring, ScoringMethod
import torch.nn.functional as F


class SelectionStrategy(ABC):
    @abstractmethod
    def select_child(self, node, min_max_stats, pruned_searchset=None):
        pass


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
        assert isinstance(node, DecisionNode)
        # assert node.expanded(), "node must be expanded to select a child"

        scores = self.scoring_method.get_scores(node, min_max_stats)

        # Masking for pruned_searchset
        if pruned_searchset is not None:
            # Create a mask for allowed actions
            # Ideally pruned_searchset should be a boolean mask tensor for efficiency
            # But if it's a list, we must convert.
            mask = torch.full_like(scores, -float("inf"))
            mask[pruned_searchset] = 0
            scores = scores + mask

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

            if pruned_searchset is not None:
                mask = torch.full_like(scores, -float("inf"))
                mask[pruned_searchset] = 0
                scores = scores + mask

            if self.temperature == 0:
                action = torch.argmax(scores).item()
            else:
                # If scores are PROBABILITIES (PriorScoring), we normalize.
                # If scores are LOGITS (Gumbel?), we softmax.
                # PriorScoring returns priors (probs).
                # GumbelScoring returns 'scores' which are shifted logits?
                # The original code assumed:
                # probs = scores / scores.sum()

                # We assume scores are strictly positive probabilities or quasi-probabilities here
                # unless logic dictates otherwise.
                # For Gumbel, we usually pick Top 1. Sampling with Gumbel is specific.

                # Safe handling:
                # values can be negative?
                # If PriorScoring, values are probs [0,1].

                probs = scores
                if self.temperature != 1.0:
                    # apply temp to probs? probs^(1/T)
                    # safeguard against negative or zero
                    probs = torch.pow(probs.clamp(min=1e-8), 1.0 / self.temperature)

                # Normalize
                probs = probs / probs.sum()

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
