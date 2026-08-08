import torch
import torch.nn.functional as F
from typing import Tuple

# TODO: should we make something like torch.Optim classes for TD optimization. I feel like we have these update rules similar to things like our IDBD or ObGD update rules and we could make an optimizer class for these or something? Or is that a bad idea?


# GRADIENT TD METHODS
# TODO: more semantic naming instead of phi and theta etc?
# TODO: actually use/test importance sampling
# TODO: batch updates where phi comes from vectorized envs. "batched online learning"
# TODO: remove value specific logic to make these work for Policy updates URGENT FOR STREAM RL.
# TODO: allow for entropy regularization with TD policy method
# TODO: is it possible to unify these?
# TODO: there is an orginization and semantic issue arising here. not all interfaces are the same. and there are some stream TD methods that use gradients to update weights (as in stream RL works) some that work only on linear methods, some that get expanded to work on linear and non linear methods with backprop. So there is a like a mix of things going on here.
# TODO: does this work with non linear weights/networks?
# TODO: should this be a function since now that we passed error instead of computing it in the function its one line.


def gtd0_update_(
    error: float | torch.Tensor,
    features: torch.Tensor,
    next_features: torch.Tensor,
    gamma: float | torch.Tensor,
    weights: torch.Tensor,
    u: torch.Tensor,
    alpha: float | torch.Tensor,
    beta: float | torch.Tensor,
    terminated: bool | torch.Tensor,
    rho: float | torch.Tensor = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GTD(0) update from Sutton et al. (2009). NOTE: Faithful to the original 2009 paper, not modern GTD2/TDC.

    Args:
        error: The scalar error term (e.g., TD error or Advantage).
        features: Feature vector of the current state [features].
        next_features: Feature vector of the next state [features].
        gamma: Discount factor.
        weights: Current weight vector [features].
        u: Auxiliary weight vector for GTD(0) [features].
        alpha: Learning rate.
        beta: Step size for auxiliary weight updates.
        terminated: Whether the next state is a terminal state.
        rho: Importance sampling ratio (default: 1.0 for on-policy).

    Returns:
        The updated weight vector weights [features] and auxiliary weight vector u [features].

    NOTE: This implementation is strictly TD(0). It does not yet support eligibility traces.
    NOTE: Strictly linear function approximation.
    """
    u_dot_feat = torch.dot(features, u)

    # Update auxiliary weights (u)
    u.add_(error * features - u, alpha=beta * rho)

    # Update primary weights (weights)
    weights.add_(
        (features - gamma * next_features * (1.0 - float(terminated))) * u_dot_feat,
        alpha=alpha * rho,
    )

    return weights, u
