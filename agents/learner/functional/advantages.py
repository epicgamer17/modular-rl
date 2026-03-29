import numpy as np
import torch
from typing import Tuple
from agents.learner.functional.returns import discounted_cumulative_sums


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalized Advantage Estimation (GAE).
    Returns (advantages, returns).
    
    Args:
        rewards: [T]
        values: [T]
        bootstrap_value: float scalar
        gamma: float
        gae_lambda: float
        
    Returns:
        advantages: [T]
        returns: [T]
    """
    values_appended = np.append(values, bootstrap_value)
    # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    deltas = rewards + gamma * values_appended[1:] - values_appended[:-1]
    
    advantages = discounted_cumulative_sums(deltas, gamma * gae_lambda)
    returns = advantages + values
    
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Normalizes advantages to have mean 0 and std 1.
    
    Args:
        advantages: [B, ...] tensor
        eps: constant for numerical stability
        
    Returns:
        normalized_advantages: [B, ...] tensor
    """
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + eps)
