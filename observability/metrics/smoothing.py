import numpy as np
from typing import List

def exponential_moving_average(values: List[float], alpha: float = 0.1) -> List[float]:
    """Compute EMA of a sequence."""
    if not values:
        return []
    
    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])
    return smoothed

def simple_moving_average(values: List[float], window: int = 10) -> List[float]:
    """Compute SMA of a sequence."""
    if len(values) < window:
        return values
    
    return np.convolve(values, np.ones(window)/window, mode='valid').tolist()
