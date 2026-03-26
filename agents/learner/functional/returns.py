import torch
from torch import Tensor
import numpy as np
from typing import Tuple, Dict, Optional


def discounted_cumulative_sums(x: np.ndarray, discount: float) -> np.ndarray:
    """
    Discounted cumulative sums for computing rewards-to-go and advantage estimates.
    Uses scipy.signal.lfilter for speed.
    """
    import scipy.signal
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]





def compute_unrolled_n_step_targets(
    raw_rewards: Tensor,
    raw_values: Tensor,
    raw_to_plays: Tensor,
    raw_terminated: Tensor,
    valid_mask: Tensor,
    gamma: float,
    n_step: int,
    unroll_steps: int,
) -> Tuple[Tensor, Tensor]:
    """
    Vectorized N-step target calculation for unrolled sequences.
    Handles multi-player two-player zero-sum value transformation (signs).
    
    Args:
        raw_rewards: [B, L] 
        raw_values: [B, L]
        raw_to_plays: [B, L]
        raw_terminated: [B, L]
        valid_mask: [B, L]
        gamma: float
        n_step: int
        unroll_steps: int
        
    Returns:
        target_values: [B, unroll_steps + 1]
        target_rewards: [B, unroll_steps + 1] (instant rewards, not prefix)
    """
    device = raw_rewards.device
    batch_size = raw_rewards.shape[0]
    num_windows = unroll_steps + 1
    required_len = num_windows + n_step - 1

    # 1. Setup Windows
    # [B, num_windows, n_step]
    def unfold(tensor, length, size):
        if tensor.dim() == 2:
            return tensor[:, :length].unfold(1, size, 1)
        return tensor[:, :length].unfold(1, size, 1)

    rewards_windows = unfold(raw_rewards, required_len, n_step)
    to_plays_windows = unfold(raw_to_plays, required_len, n_step)
    terminated_windows = unfold(raw_terminated, required_len, n_step)
    valid_windows = unfold(valid_mask, required_len, n_step)

    # 2. Compute Segmented Sums
    gammas = (
        gamma ** torch.arange(n_step, dtype=torch.float32, device=device)
    ).reshape(1, 1, n_step)

    # Calculate "was done before this step in the window"
    dones_float = terminated_windows.float()
    was_done_before = torch.cat(
        [
            torch.zeros((batch_size, num_windows, 1), device=device),
            torch.cumsum(dones_float, dim=2)[:, :, :-1],
        ],
        dim=2,
    )

    valid_steps_mask = valid_windows & (was_done_before == 0)
    
    # Signs for zero-sum value relative to window start
    current_to_plays = raw_to_plays[:, :num_windows].unsqueeze(2)
    signs = torch.where(current_to_plays == to_plays_windows, 1.0, -1.0)
    
    weighted_rewards = rewards_windows * gammas * signs * valid_steps_mask.float()
    summed_rewards = weighted_rewards.sum(dim=2)

    # 3. Compute Bootstrap Value
    boot_indices = torch.arange(n_step, n_step + num_windows, device=device)
    safe_boot_indices = torch.clamp(boot_indices, max=raw_values.shape[1] - 1)

    boot_values = raw_values[:, safe_boot_indices]
    boot_to_plays = raw_to_plays[:, safe_boot_indices]
    
    # Bootstrap is valid if: same game AND window didn't terminate AND step itself isn't terminal
    hit_terminated_in_window = (
        terminated_windows.float() * valid_steps_mask.float()
    ).sum(dim=2) > 0
    
    boot_is_valid = (
        valid_mask[:, safe_boot_indices]
        & (~hit_terminated_in_window)
        & (~raw_terminated[:, safe_boot_indices])
    )

    boot_signs = torch.where(
        raw_to_plays[:, :num_windows] == boot_to_plays, 1.0, -1.0
    )
    boot_term = (gamma**n_step) * boot_values * boot_signs

    target_values = summed_rewards + torch.where(
        boot_is_valid, boot_term, torch.tensor(0.0, device=device)
    )
    
    # Grounding: Past game end = 0.0
    target_values = target_values * valid_mask[:, :num_windows].float()

    # 4. Filter Instant Rewards
    target_rewards = torch.zeros(
        (batch_size, num_windows), dtype=torch.float32, device=device
    )
    t_rew = raw_rewards[:, : num_windows - 1]
    mask_slice = valid_mask[:, : num_windows - 1]
    
    target_slice = torch.zeros_like(target_rewards[:, 1:])
    target_slice[mask_slice] = t_rew[mask_slice]
    target_rewards[:, 1:] = target_slice

    return target_values, target_rewards
