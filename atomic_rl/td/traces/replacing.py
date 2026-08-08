import torch


def compute_replacing_traces(
    traces: torch.Tensor,
    features: torch.Tensor,  # Can be continuous/non-binary
    gamma: float | torch.Tensor,
    lam: float | torch.Tensor,
    terminated: torch.Tensor,
) -> torch.Tensor:
    """
    Updates eligibility traces using the replacing trace method (Sutton & Barto).
    Formula: e_t = max(gamma * lambda * e_{t-1}, phi_t)

    NOTE: Replacing traces are usually defined only for discrete states or linear function approximation with binary features (that are either 1 or 0, present or not present)

    Extended in True Online TD(lambda) to handle non-binary features as follows:
    e_{i,t} = γλe_{i,t−1} if φ_{i,t} = 0
            = αφ_{i,t} if φ_{i,t} != 0

    TODO: possible future work True Online TD Lambda for offline case (is there a paper for this?)
    """
    # Expand terminated to match feature dimensions [B, 1]
    term_mask = terminated.unsqueeze(-1).float()

    # 1. Calculate the standard decayed trace (resetting if terminated)
    decayed_traces = gamma * lam * traces * (1.0 - term_mask)

    # 2. Apply the conditional replacement
    # Using a small epsilon or exact 0 check depending on your feature precision
    feature_is_zero = features == 0.0

    # If the feature is 0, keep the decayed trace. Otherwise, REPLACE it.
    new_traces = torch.where(feature_is_zero, decayed_traces, features)

    return new_traces
