import torch


def compute_true_online_traces(
    traces: torch.Tensor,  # [B, features]
    features: torch.Tensor,  # [B, features]
    alpha: float | torch.Tensor,
    gamma: float | torch.Tensor,
    lam: float | torch.Tensor,
    terminated: torch.Tensor,  # [B]
) -> torch.Tensor:
    """
    Updates eligibility traces using the True Online TD(lambda) method (Sutton 2014).
    Formula: e_t = gamma * lambda * e_{t-1} + alpha * (1 - gamma * lambda * e_{t-1}^T features_t) * features_t

    Args:
        traces: The eligibility traces from the previous step.
        features: The feature vector of the current state.
        alpha: Learning rate.
        gamma: Discount factor.
        lam: Trace decay rate (lambda).
        terminated: Mask [B] indicating episode termination to clear traces.

    Returns:
        The updated traces of shape [batch, num_features].

    NOTE: We implement True Online TD(lambda) trace update from Suttons Textbook (2nd Ed.) not from the True Online TD(lambda) paper.
    """
    # Fail Fast: Ensure shape alignment
    assert (
        traces.shape == features.shape
    ), f"Trace {traces.shape} and feature {features.shape} shapes must match"
    assert (
        terminated.ndim == 1
    ), f"Expected 1D terminated tensor [B], got {terminated.shape}"

    term_mask = terminated.unsqueeze(-1).float()

    # \gamma * \lambda * z
    z_decay = gamma * lam * traces * (1.0 - term_mask)

    # z^T * x
    inner_dot = torch.sum(traces * features, dim=-1, keepdim=True)

    # z_t = gamma * lambda * z_{t-1} + (1 - alpha * gamma * lambda * z_{t-1}^T features_t) * features_t
    new_traces = z_decay + (1.0 - alpha * gamma * lam * inner_dot) * features

    return new_traces
