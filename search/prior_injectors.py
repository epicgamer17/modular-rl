from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

import numpy as np
import torch

from modules.utils import support_to_scalar
from search.initial_searchsets import SearchSet, SelectAll, SelectTopK
from utils.utils import get_legal_moves


class PriorInjector(ABC):
    @abstractmethod
    def inject(self, policy, config, trajectory_action=None):
        """Modifies the context (policy, scores, etc.) in place."""
        pass


class DirichletInjector(PriorInjector):
    def inject(self, policy, legal_moves, config, trajectory_action=None):
        # Only apply noise to legal moves
        if config.root_dirichlet_alpha_adaptive:
            alpha = 1.0 / np.sqrt(len(legal_moves))
        else:
            alpha = config.root_dirichlet_alpha

        noise = np.random.dirichlet([alpha] * len(legal_moves))
        frac = config.root_exploration_fraction

        # Map noise back to the full policy tensor (or just relevant indices)
        # Note: We operate on the policy probabilities
        new_policy = policy.clone()
        for i, action in enumerate(legal_moves):
            new_policy[action] = (1 - frac) * policy[action] + frac * noise[i]

        return new_policy


class ActionTargetInjector(PriorInjector):
    """
    Corresponds to the logic in the original ActionInjectionStrategy.
    Boosts the prior of the trajectory_action.
    """

    def inject(self, policy, legal_moves, config, trajectory_action=None):
        # TODO: a clean way of properly ensuring policy is masked here using legal moves
        if trajectory_action is None:
            return policy

        # Sanity check: user must ensure trajectory_action is legal/possible if filtering
        # Ideally, this injector runs after a selector that ensures the action is present,
        # but here we modify priors before selection to ensure it gets picked if using TopK.

        inject_frac = config.injection_frac

        # Calculate total mass to normalize existing priors
        # We sum over legal moves or all moves depending on how policy is masked.
        # Assuming policy is valid over legal moves.
        total_prior = torch.sum(policy).item()

        # Renormalize priors: put (1-inject_frac) of current mass on existing priors
        policy = (1.0 - inject_frac) * (policy / total_prior)

        # Boost injected action
        policy[trajectory_action] += inject_frac
        return policy


class GumbelInjector(PriorInjector):
    """
    Injects Gumbel noise into the logits, used for Gumbel MuZero selection.
    """

    def inject(self, policy, legal_moves, config, trajectory_action=None):
        assert legal_moves and len(legal_moves) > 0
        # Gumbel noise: g = -log(-log(uniform))
        g = -torch.log(-torch.log(torch.rand(len(legal_moves))))

        # Update scores: Score = g + logits
        # We map these back to the full actions space in context.scores
        # TODO: MUST STORE NETWORK PRIOR AND PRIOR IS ESSENTIALLY PRIOR SCORE NOT NETWORK PRIOR
        logits = torch.log(policy + 1e-12).cpu()
        current_logits = logits[legal_moves]
        noisy_scores = g + current_logits

        new_logits = logits.clone()
        new_logits[legal_moves] = noisy_scores

        # TODO: RETURN NOISY SCORES POLICY, TURN LOGITS INTO POLICY, IS BELOW RIGHT?
        assert not torch.all(torch.isclose(new_logits, logits))
        return torch.softmax(new_logits, dim=-1)
