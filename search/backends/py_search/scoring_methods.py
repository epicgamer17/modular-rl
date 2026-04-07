import torch
import torch.nn.functional as F
from math import sqrt, log
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass

from search.backends.py_search.nodes import ChanceNode, DecisionNode
from search.backends.py_search.utils import get_completed_q_improved_policy


# --- Scoring Methods ---


class ScoringMethod(ABC):
    @abstractmethod
    def score(self, node, child, min_max_stats) -> float:
        """Returns a score for a single child."""
        pass  # pragma: no cover

    def get_scores(self, node, min_max_stats) -> Dict[int, float]:
        """
        Returns scores for all children of the node.
        Can be overridden for efficiency (e.g., batch computations like Gumbel).
        """
        scores = {}
        for action, child in node.children.items():
            scores[action] = self.score(node, child, min_max_stats)
        return scores

    def score_initial(self, prior: float, action: int) -> float:
        """
        Returns a score based solely on the prior (for use in SelectTopK or pruning before expansion).
        Default implementation returns the prior itself.
        """
        return prior


class UCBScoring(ScoringMethod):
    def __init__(self, bootstrap_method: str = "parent_value"):
        self.bootstrap_method = bootstrap_method

    def score(self, node, child, min_max_stats) -> float:
        # Fallback to single score if needed
        raise NotImplementedError(
            "Use get_scores for vectorized UCB"
        )  # pragma: no cover

    def get_scores(self, node, min_max_stats) -> torch.Tensor:
        # Vectorized UCB
        # Use math functions for scalar part
        pb_c_val = (
            log((node.visits + node.pb_c_base + 1) / node.pb_c_base) + node.pb_c_init
        )
        pb_c_val *= sqrt(node.visits)

        # Broadcast to tensor
        pb_c = torch.tensor(
            pb_c_val, device=node.child_visits.device, dtype=torch.float32
        )
        pb_c = pb_c / (node.child_visits + 1)

        prior_score = pb_c * node.child_priors

        # Value score
        # We need Q-values for all children.
        # node.child_values contains Q-values for visited children.
        # For unvisited children (visits==0), we need to bootstrap.

        # Identify visited nodes
        visited_mask = node.child_visits > 0

        # Initialize q_values with child_values (correct for visited)
        q_values = node.child_values.clone()

        # For unvisited, use bootstrap value.
        if not visited_mask.all():
            if self.bootstrap_method == "parent_value":
                bootstrap_val = node.value()
            elif self.bootstrap_method == "zero":
                bootstrap_val = 0.0
            elif self.bootstrap_method == "v_mix":
                bootstrap_val = node.get_v_mix()
            elif self.bootstrap_method == "mu_fpu":
                total_vis = node.child_visits.sum()
                if total_vis > 0:
                    bootstrap_val = (
                        node.child_values * node.child_visits
                    ).sum() / total_vis
                else:
                    bootstrap_val = node.value()
            else:
                bootstrap_val = node.value()

            q_values[~visited_mask] = bootstrap_val

        # Normalize Q-values
        # Standard normalization across ALL values (bootstrapped included)
        value_score = min_max_stats.normalize(q_values)

        # Combine
        scores = prior_score + value_score
        return scores


class GumbelScoring(ScoringMethod):
    def __init__(self, gumbel_cvisit: int, gumbel_cscale: float):
        self.gumbel_cvisit = gumbel_cvisit
        self.gumbel_cscale = gumbel_cscale

    def score(self, node, child, min_max_stats) -> float:
        raise NotImplementedError(
            "Use get_scores for vectorized Gumbel"
        )  # pragma: no cover

    def get_scores(self, node, min_max_stats) -> torch.Tensor:
        # Vectorized Gumbel
        pi0 = get_completed_q_improved_policy(
            self.gumbel_cvisit, self.gumbel_cscale, node, min_max_stats
        )

        visits = node.child_visits
        sum_N = visits.sum()
        denom = 1.0 + sum_N

        scores = pi0 - (visits / denom)
        return scores

    def score_initial(self, prior: float, action: int) -> float:
        # For Gumbel, initial scoring (before expansion) is often g + logits
        # Here we approximate or just return prior if g is not available in this context
        # But commonly TopK for Gumbel uses raw priors (logits) or g+logits.
        # Assuming SelectTopK passes 'prior' which is a probability.
        return prior


class LeastVisitedScoring(ScoringMethod):
    def score(self, node, child, min_max_stats) -> float:
        return -float(child.visits)

    def get_scores(self, node, min_max_stats) -> torch.Tensor:
        return -node.child_visits


class PriorScoring(ScoringMethod):
    """Simple scoring based on priors."""

    def score(self, node, child, min_max_stats) -> float:
        p = child.prior
        return p.item() if torch.is_tensor(p) else p

    def score_initial(self, prior: float, action: int) -> float:
        return prior

    def get_scores(self, node, min_max_stats) -> torch.Tensor:
        return node.child_priors


class QValueScoring(ScoringMethod):
    """
    Scores nodes based on their Q-value.
    """

    def score(self, node, child, min_max_stats) -> float:
        if child.expanded():
            v = node.get_child_q_from_parent(child)
        else:
            v = child.value()
        return v.item() if torch.is_tensor(v) else v

    def get_scores(self, node, min_max_stats) -> torch.Tensor:
        # TODO: handle unvisited nodes?
        # visited mask?
        # If unvisited, child_values is 0.
        # Standard Q-value scoring might want bootstrap?
        # But this is usually used for BestActionRootPolicy which looks for valid best action.
        # If we use raw child_values, unvisited (0) might be selected if others are negative.
        # But generally Q-values are used when we have visited enough.
        return node.child_values

    def score_initial(self, prior: float, action: int) -> float:
        return prior


class DeterministicChanceScoring(ScoringMethod):
    """
    Scores chance nodes deterministically based on prior / (visits + 1).
    Used to select the most probable under-explored outcome.
    """

    def score(self, node, child, min_max_stats) -> float:
        raise NotImplementedError("Use get_scores for vectorized")  # pragma: no cover

    def get_scores(self, node, min_max_stats) -> torch.Tensor:
        return node.child_priors / (node.child_visits + 1.0)

    def score_initial(self, prior: float, action: int) -> float:
        return prior
