from abc import ABC, abstractmethod
import torch
import numpy as np

from search.utils import (
    calculate_gumbel_sigma,
    get_completed_q,
    get_completed_q_improved_policy,
)

from search.scoring_methods import QValueScoring


class RootPolicyStrategy(ABC):
    def __init__(self, config, device, num_actions):
        self.config = config
        self.device = device
        self.num_actions = num_actions

    @abstractmethod
    def get_policy(self, root, min_max_stats):
        """
        Returns the final policy distribution (probability vector)
        over all actions based on the search results.
        Returns: torch.Tensor of shape (num_actions,)
        """
        pass


class VisitFrequencyPolicy(RootPolicyStrategy):
    """
    Standard AlphaZero Policy:
    Returns probabilities proportional to the visit counts of the children.
    Includes temperature support (visits ^ (1/T)).
    """

    def __init__(self, config, device, num_actions, temperature=1.0):
        super().__init__(config, device, num_actions)
        self.temperature = temperature

    def get_policy(self, root, min_max_stats):
        # Gather visits
        # Vectorized access
        visits = root.child_visits.to(self.device)

        if self.temperature == 0:
            # Greedy Argmax (break ties randomly or first)
            # Create a one-hot distribution on the max visited action
            max_visit = torch.max(visits)
            # Mask for all actions that share the max visit count
            mask = (visits == max_visit).float()
            # Normalize to handle ties
            policy = mask / mask.sum()
            return policy

        elif self.temperature == 1.0:
            # Standard Proportional
            sum_visits = torch.sum(visits)
            assert sum_visits > 0
            return visits / sum_visits
        else:
            # Temperature Adjusted
            # prevent numerical instability with 0 visits by adding epsilon or masking
            # Clamp to avoid 0^neg_power issue?
            # Or just visits^(1/T)
            visits_powered = torch.pow(visits, 1.0 / self.temperature)
            sum_powered = torch.sum(visits_powered)
            assert sum_powered > 0
            return visits_powered / sum_powered


class CompletedQValuesRootPolicy(RootPolicyStrategy):
    """
    Gumbel MuZero Policy:
    Calculates the 'Improved Policy' (pi0) using the Completed Q-values and Sigma transformation.
    This is distinct from visit counts and is the mathematically correct policy target for Gumbel MuZero.
    """

    def get_policy(self, root, min_max_stats):
        return get_completed_q_improved_policy(self.config, root, min_max_stats)


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
