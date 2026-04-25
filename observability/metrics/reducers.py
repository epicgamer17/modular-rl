import numpy as np
from typing import List, Any

def mean(values: List[Any]) -> float:
    return float(np.mean(values))

def std(values: List[Any]) -> float:
    return float(np.std(values))

def min_val(values: List[Any]) -> float:
    return float(np.min(values))

def max_val(values: List[Any]) -> float:
    return float(np.max(values))

def last(values: List[Any]) -> Any:
    return values[-1] if values else None
