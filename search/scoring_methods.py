import torch
import torch.nn.functional as F
import numpy as np
from math import sqrt, log
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass

from search.nodes import ChanceNode, DecisionNode
from search.utils import get_completed_q_improved_policy


# --- Scoring Methods ---


class ScoringMethod(ABC):
    @abstractmethod
    def score(self, node, child, min_max_stats) -> float:
        """Returns a score for a single child."""
        pass

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
    def score(self, node, child, min_max_stats) -> float:
        pb_c = log((node.visits + node.pb_c_base + 1) / node.pb_c_base) + node.pb_c_init
        pb_c *= sqrt(node.visits) / (child.visits + 1)
        prior_score = pb_c * child.prior

        if child.expanded():
            value_score = min_max_stats.normalize(node.get_child_q_from_parent(child))
        else:
            value_score = min_max_stats.normalize(child.value())

        # check if value_score is nan
        assert value_score == value_score, "value_score is nan"
        assert prior_score == prior_score, "prior_score is nan"

        score = prior_score + value_score
        return score.detach().item() if torch.is_tensor(score) else score


class GumbelScoring(ScoringMethod):
    def __init__(self, config):
        self.config = config

    def score(self, node, child, min_max_stats) -> float:
        # Note: This is inefficient if called individually for all children.
        # Use get_scores for batch computation.
        scores = self.get_scores(node, min_max_stats)
        # Find the action corresponding to this child
        for action, ch in node.children.items():
            if ch is child:
                return scores[action]
        raise ValueError("Child not found in node's children")

    def get_scores(self, node, min_max_stats) -> Dict[int, float]:
        pi0 = get_completed_q_improved_policy(self.config, node, min_max_stats)

        # Actions in node.children might be a subset of all actions if filtered,
        # but pi0 covers all actions in network_policy (usually).
        # We only care about actions present in node.children
        scores = {}
        for action, child in node.children.items():
            # compute selection metric: pi0(a) - N(a) / (1 + sum_N)
            visits = float(child.visits)

            # Helper to get sum_N over the specific actions available in children
            # (Matches original implementation logic which summed visits of candidate actions)
            all_visits = torch.tensor(
                [float(ch.visits) for ch in node.children.values()]
            )
            sum_N = float(all_visits.sum())
            denom = 1.0 + sum_N

            # pi0 is a tensor over all actions, index with action
            score = pi0[action].item() - (visits / denom)
            scores[action] = score

        return scores

    def score_initial(self, prior: float, action: int) -> float:
        # For Gumbel, initial scoring (before expansion) is often g + logits
        # Here we approximate or just return prior if g is not available in this context
        # But commonly TopK for Gumbel uses raw priors (logits) or g+logits.
        # Assuming SelectTopK passes 'prior' which is a probability.
        return prior


class LeastVisitedScoring(ScoringMethod):
    def score(self, node, child, min_max_stats) -> float:
        # We want least visited, so score is negative visits
        return -float(child.visits)


class PriorScoring(ScoringMethod):
    """Simple scoring based on priors."""

    def score(self, node, child, min_max_stats) -> float:
        p = child.prior
        return p.detach().item() if torch.is_tensor(p) else p

    def score_initial(self, prior: float, action: int) -> float:
        return prior


class QValueScoring(ScoringMethod):
    """
    Scores nodes based on their Q-value.
    """

    def score(self, node, child, min_max_stats) -> float:
        if child.expanded():
            v = node.get_child_q_from_parent(child)
        else:
            v = child.value()
        return v.detach().item() if torch.is_tensor(v) else v

    def score_initial(self, prior: float, action: int) -> float:
        return prior
