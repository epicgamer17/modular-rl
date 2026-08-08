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


def semi_gradient_td_update_(
    error: float | torch.Tensor,
    weights: torch.Tensor,
    alpha: float | torch.Tensor,
    update_vector: torch.Tensor,
    rho: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """
    Performs a generic semi-gradient update for linear function approximation. Allows for eligibility traces.
    Can be used for both value functions (where error is TD error) and policies (where error is advantage).

    Args:
        error: The scalar error term (e.g., TD error or Advantage).
        weights: Current weight vector [features].
        alpha: Learning rate.
        update_vector: The vector used to step the weights.
            - For TD(0) value update, pass `features`.
            - For TD(lambda) value update, pass the accumulated `eligibility_trace`.
            - For Policy update, pass the policy gradient or its trace.
        rho: Importance sampling ratio (default: 1.0 for on-policy).

    Returns:
        The updated weight vector weights [features].

    NOTE: Strictly linear function approximation.
    """
    weights.add_(update_vector, alpha=alpha * rho * error)
    return weights
