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


# TODO: should v_next be handled here?
def true_online_td_update_(
    error: float | torch.Tensor,
    v_current: float | torch.Tensor,
    v_old: float | torch.Tensor,
    features: torch.Tensor,
    weights: torch.Tensor,
    alpha: float | torch.Tensor,
    trace: torch.Tensor,
) -> torch.Tensor:
    """
    Performs a True Online Temporal Difference (TD) update for linear function approximation.

    Args:
        error: The scalar error term (e.g., TD error or Advantage).
        v_current: The value of the current state computed using the current weight vector.
        v_old: The value of the current state computed using the previous weight vector.
        features: Feature vector of the current state [features].
        weights: Current weight vector [features].
        alpha: Learning rate.
        trace: The updated True Online eligibility trace for the current step (e_t) [features].

    Returns:
        weights: The updated weight vector [features].

    NOTE: Strictly linear function approximation.
    NOTE: We implement True Online TD(lambda) weight update from Suttons Textbook (2nd Ed.) not from the True Online TD(lambda) paper.
    """
    # Fail Fast: Ensure shape alignment
    assert features.ndim == 1, f"Expected 1D features [features], got {features.shape}"
    assert (
        weights.shape == features.shape
    ), f"Shape mismatch: weights {weights.shape}, features {features.shape}"

    # w <- w + \alpha * (\delta + V - V_old) * z - \alpha * (V - V_old) * x
    v_diff = v_current - v_old
    weights.add_(trace, alpha=alpha * (error + v_diff))
    weights.sub_(features, alpha=alpha * v_diff)

    return weights
