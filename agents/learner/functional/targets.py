import torch
from torch import Tensor
from typing import Tuple, Dict, Any


def compute_td_target(
    rewards: Tensor,
    dones: Tensor,
    next_values: Tensor,
    gamma: float,
    n_step: int = 1,
) -> Tensor:
    """
    Standard TD target calculation: r + gamma^n * (1-d) * V(s').
    
    Args:
        rewards: vector of rewards
        dones: vector of terminal flags
        next_values: predicted values of next states
        gamma: discount factor
        n_step: number of steps in the n-step sequence
        
    Returns:
        TD target for training.
    """
    discount = gamma ** n_step
    # bootstrap_on_truncated should be handled before calling this by passing processed dones.
    return rewards + (1.0 - dones.float()) * discount * next_values


def project_onto_grid(
    shifted_support: Tensor,
    probabilities: Tensor,
    vmin: float,
    vmax: float,
    bins: int,
) -> Tensor:
    """
    Pure geometric projection for C51/Distributional RL mass.
    Snaps an arbitrary distribution onto a fixed grid.
    
    Args:
        shifted_support: [B, Atoms] Locations of the mass after Bellman shift.
        probabilities: [B, Atoms] Probability mass at each location.
        vmin: grid min
        vmax: grid max
        bins: number of atoms in grid
        
    Returns:
        [B, Atoms] mass on fixed grid support.
    """
    # 1. Capture original shape and flatten to [N, Atoms]
    orig_shape = shifted_support.shape
    N = shifted_support.numel() // bins
    z = shifted_support.reshape(N, bins).clamp(vmin, vmax)
    p = probabilities.reshape(N, bins)
    
    delta_z = (vmax - vmin) / (bins - 1)
    device = z.device

    # 2. Calculate offsets and bounds (l, u)
    b = (z - vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    # 3. Calculate mass distribution
    p_u = p * (b - l.float())
    p_l = p * (1.0 - (b - l.float()))

    # 4. Safe indexing for scattering
    l_idx = l.clamp(0, bins - 1)
    u_idx = u.clamp(0, bins - 1)

    projected = torch.zeros((N, bins), device=device, dtype=p.dtype)
    projected.scatter_add_(1, l_idx, p_l)
    projected.scatter_add_(1, u_idx, p_u)

    # 5. Return unflattened
    return projected.reshape(*orig_shape)


def compute_c51_target(
    rewards: Tensor,
    next_probs: Tensor,
    support: Tensor,
    dones: Tensor,
    gamma: float,
    n_step: int = 1,
) -> Tensor:
    """
    Bellman shift and projection for C51/Distributional RL.
    
    Args:
        rewards: [B]
        next_probs: [B, Atoms] selected next action state distribution
        support: [Atoms] fixed grid values (e.g. linspace(vmin, vmax, Atoms))
        dones: [B] terminal flags
        gamma: float
        n_step: int
        
    Returns:
        Categorical probability distribution on fixed support.
    """
    discount = gamma ** n_step
    vmin = support[0].item()
    vmax = support[-1].item()
    bins = support.numel()

    # 1. Shift: [B, 1] + [B, 1] * [1, Atoms] -> [B, Atoms]
    shifted_support = rewards.unsqueeze(1) + discount * (
        1.0 - dones.float()
    ).unsqueeze(1) * support.unsqueeze(0)

    # 2. Project
    return project_onto_grid(shifted_support, next_probs, vmin, vmax, bins)
