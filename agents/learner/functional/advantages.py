import numpy as np
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
