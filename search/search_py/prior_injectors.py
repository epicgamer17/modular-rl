from abc import ABC, abstractmethod
from typing import Optional, Any, List

import numpy as np
import torch


from .utils import _safe_log_probs


class PriorInjector(ABC):
    @abstractmethod
    def inject(
        self,
        policy: torch.Tensor,
        legal_moves: List[int],
        trajectory_action: Optional[int] = None,
        policy_dist: Optional[Any] = None,
        exploration: bool = True,
    ) -> torch.Tensor:
        """Modifies the context (policy, scores, etc.) in place."""
        pass  # pragma: no cover


class DirichletInjector(PriorInjector):
    def __init__(self, use_dirichlet: bool, dirichlet_alpha: float, dirichlet_fraction: float):
        self.use_dirichlet = use_dirichlet
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_fraction = dirichlet_fraction

    def inject(
        self,
        policy: torch.Tensor,
        legal_moves: List[int],
        trajectory_action: Optional[int] = None,
        policy_dist: Optional[Any] = None,
        exploration: bool = True,
    ) -> torch.Tensor:
        if not exploration or not self.use_dirichlet:
            return policy

        alpha = self.dirichlet_alpha
        noise = np.random.dirichlet([alpha] * len(legal_moves))
        frac = self.dirichlet_fraction

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

    def __init__(self, injection_frac: float):
        self.injection_frac = injection_frac

    def inject(
        self,
        policy: torch.Tensor,
        legal_moves: List[int],
        trajectory_action: Optional[int] = None,
        policy_dist: Optional[Any] = None,
        exploration: bool = True,
    ) -> torch.Tensor:
        # Note: Action injection is usually for re-analysis, not stochastic exploration.
        # We keep it as is regardless of exploration flag unless explicitly asked.
        if trajectory_action is None:
            return policy

        # Sanity check: user must ensure trajectory_action is legal/possible if filtering
        # Ideally, this injector runs after a selector that ensures the action is present,
        # but here we modify priors before selection to ensure it gets picked if using TopK.

        inject_frac = self.injection_frac

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

    def inject(
        self,
        policy: torch.Tensor,
        legal_moves: List[int],
        trajectory_action: Optional[int] = None,
        policy_dist: Optional[Any] = None,
        exploration: bool = True,
    ) -> torch.Tensor:
        if not exploration:
            return policy

        assert legal_moves and len(legal_moves) > 0

        # Prefer distribution logits (from model output/masked dist) to avoid
        # reconstructing logits from probs with epsilon hacks.
        logits = None
        if policy_dist is not None and hasattr(policy_dist, "logits"):
            logits = policy_dist.logits
            if logits is not None:
                logits = logits.cpu()
        if logits is None:
            logits = _safe_log_probs(policy).cpu()
        if logits.dim() > 1:
            if logits.shape[0] != 1:
                raise ValueError("GumbelInjector expects a single policy vector.")
            logits = logits[0]

        # Keep illegal actions exactly masked.
        legal_idx = torch.as_tensor(legal_moves, dtype=torch.long)
        illegal_mask = torch.ones_like(logits, dtype=torch.bool)
        illegal_mask[legal_idx] = False
        logits = logits.clone()
        logits[illegal_mask] = -float("inf")

        # Gumbel noise: g = -log(-log(uniform))
        uniform_noise = torch.rand(len(legal_moves), dtype=logits.dtype).clamp(
            min=1e-8, max=1 - 1e-8
        )
        g = -torch.log(-torch.log(uniform_noise))

        # Update scores: Score = g + logits
        # We map these back to the full actions space in context.scores
        # TODO: MUST STORE NETWORK PRIOR AND PRIOR IS ESSENTIALLY PRIOR SCORE NOT NETWORK PRIOR
        current_logits = logits[legal_idx]
        noisy_scores = g + current_logits

        new_logits = logits.clone()
        new_logits[legal_idx] = noisy_scores

        # TODO: RETURN NOISY SCORES POLICY, TURN LOGITS INTO POLICY, IS BELOW RIGHT?
        assert not torch.all(torch.isclose(new_logits, logits))
        return torch.softmax(new_logits, dim=-1)
