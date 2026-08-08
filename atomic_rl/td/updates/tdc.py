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


def tdc_update_(
    error: float | torch.Tensor,
    features: torch.Tensor,
    next_features: torch.Tensor,
    gamma: float | torch.Tensor,
    weights: torch.Tensor,
    w: torch.Tensor,
    alpha: float | torch.Tensor,
    beta: float | torch.Tensor,
    terminated: bool | torch.Tensor,
    rho: float | torch.Tensor = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast-GTD / TDC update from Sutton et al. (2009).

    Args:
        error: The scalar error term (e.g., TD error or Advantage).
        features: Feature vector of the current state [features].
        next_features: Feature vector of the next state [features].
        gamma: Discount factor.
        weights: Current weight vector [features].
        w: Auxiliary weight vector for TDC [features].
        alpha: Learning rate.
        beta: Step size for auxiliary weight updates.
        terminated: Whether the next state is a terminal state.
        rho: Importance sampling ratio (default: 1.0 for on-policy).

    Returns:
        The updated weight vector weights [features] and auxiliary weight vector w [features].

    NOTE: This implementation is strictly TD(0). It does not yet support eligibility traces.
    NOTE: Strictly linear function approximation.
    """
    w_dot_feat = torch.dot(w, features)

    # Update auxiliary weights (w)
    w.add_(features, alpha=beta * rho * (error - w_dot_feat))

    # Update primary weights (weights) with gradient correction
    weights.add_(features, alpha=alpha * rho * error)
    weights.sub_(
        next_features,
        alpha=alpha * rho * gamma * (1.0 - float(terminated)) * w_dot_feat,
    )

    return weights, w
