import torch


def compute_accumulating_traces(
    traces: torch.Tensor,  # [batch, num_features]
    gradients: torch.Tensor,  # [batch, num_features]
    gamma: float | torch.Tensor,
    lam: float | torch.Tensor,
    terminated: torch.Tensor,  # [batch]
) -> torch.Tensor:
    """
    Updates eligibility traces using the accumulating trace method.
    Formula: e_t = gamma * lambda * e_{t-1} + grad_V(s_t)

    If the episode terminates, the trace is reset to zero for that batch element.

    Args:
        traces: The eligibility traces from the previous step.
        gradients: The gradient of the value function with respect to weights (phi_t).
        gamma: Discount factor.
        lam: Trace decay rate (lambda).
        terminated: Mask [B] indicating episode termination to clear traces.

    Returns:
        The updated traces of shape [batch, num_features].

    NOTE: can be instable as the traces can grow arbitrarily large. A good update rule is needed.
    NOTE: Can be used for both value and policy traces.
    NOTE: Entropy regularization for the policy case should be done by passing it in as the gradient argument, . (from Appendix E: log_prob + tau * sign(delta) * entropy). This does mean that the trace must be updated after the TD error is calculated.
    """
    assert traces.shape == gradients.shape, "Trace and gradient shapes must match"

    # Expand terminated to match feature dimensions [B, 1] for broadcasting
    term_mask = terminated.unsqueeze(-1).float()

    # Reset trace if terminated, otherwise decay and accumulate
    new_traces = (gamma * lam * traces * (1.0 - term_mask)) + gradients
    return new_traces
