from abc import ABC, abstractmethod
import torch

from search.utils import (
    calculate_gumbel_sigma,
    get_completed_q,
    get_completed_q_improved_policy,
)

from search.scoring_methods import QValueScoring


class RootPolicyStrategy(ABC):
    def __init__(self, device: torch.device, num_actions: int):
        self.device = device
        self.num_actions = num_actions

    @abstractmethod
    def get_policy(self, root, min_max_stats):
        """
        Returns the final policy distribution (probability vector)
        over all actions based on the search results.
        Returns: torch.Tensor of shape (num_actions,)
        """
        pass  # pragma: no cover


class VisitFrequencyPolicy(RootPolicyStrategy):
    """
    Standard AlphaZero Policy:
    Returns probabilities proportional to the visit counts of the children.
    """

    def get_policy(self, root, min_max_stats):
        # Gather visits
        # Vectorized access
        visits = root.child_visits.to(self.device)

        # Standard Proportional
        sum_visits = torch.sum(visits)
        assert sum_visits > 0
        return visits / sum_visits


class CompletedQValuesRootPolicy(RootPolicyStrategy):
    """
    Gumbel MuZero Policy:
    Calculates the 'Improved Policy' (pi0) using the Completed Q-values and Sigma transformation.
    This is distinct from visit counts and is the mathematically correct policy target for Gumbel MuZero.
    """

    def __init__(
        self, device: torch.device, num_actions: int, gumbel_cvisit: int, gumbel_cscale: float
    ):
        super().__init__(device, num_actions)
        self.gumbel_cvisit = gumbel_cvisit
        self.gumbel_cscale = gumbel_cscale

    def get_policy(self, root, min_max_stats):
        return get_completed_q_improved_policy(
            self.gumbel_cvisit, self.gumbel_cscale, root, min_max_stats
        )


class BestActionRootPolicy(RootPolicyStrategy):
    """
    Returns a policy selecting the action with the highest Q-value.
    """

    def get_policy(self, root, min_max_stats):
        scorer = QValueScoring()
        # Use vectorized scores
        scores = scorer.get_scores(root, min_max_stats)

        # We need to handle potential -inf or unvisited if QValueScoring returns that?
        # Assuming QValueScoring returns valid tensors (bootstrap or 0 for unvisited).

        best_action = torch.argmax(scores).item()

        policy = torch.zeros(self.num_actions, device=self.device)
        policy[best_action] = 1.0

        return policy
