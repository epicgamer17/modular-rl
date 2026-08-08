import torch


# TODO: I think truncation right now works for single env trajectories without auto resetting, but would fail for vectorized envs with auto resetting. as the next_q_values would be on the resetted state instead of the one from the info. verify this, and if it is an issue, unify the vectorization logic in our buffer to work with offline buffers as well somehow.
def compute_v_td_target(
    next_values: torch.Tensor,  # [B]
    rewards: torch.Tensor,  # [B]
    terminated: torch.Tensor,  # [B]
    gamma: torch.Tensor,  # [B] or scalar
) -> torch.Tensor:
    """
    Calculates the 1-step Temporal Difference target for state values V(s).
    Formula: y = R_{t} + gamma * (1 - terminated) * V(s_{t+1})

    Args:
        next_values: Value estimates of the next states.
        rewards: Rewards for the transitions.
        terminated: Booleans indicating whether the episodes terminated.
        gamma: Discount factors.

    Returns:
        torch.Tensor: The detached TD target of shape [B]. Gradients will NOT flow through this tensor.
    """
    # Fail Fast: Ensure shape alignment
    assert (
        next_values.ndim == 1
    ), f"Expected 1D next_values [B], got {next_values.shape}"
    assert (
        rewards.shape == terminated.shape == next_values.shape
    ), f"Shape mismatch: rewards {rewards.shape}, terminated {terminated.shape}, next_values {next_values.shape}"

    target = rewards + gamma * next_values * (1 - terminated.float())
    return target.detach()
