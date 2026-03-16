import collections
from typing import Optional, List

MAXIMUM_FLOAT_VALUE = float("inf")

# TODO: EFFICIENT ZERO SOFT MINMAX STATS


class MinMaxStats(object):
    def __init__(
        self,
        known_bounds: Optional[List[float]],
        epsilon: float = 1e-8,
    ):
        self.max = known_bounds[1] if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.min = known_bounds[0] if known_bounds else MAXIMUM_FLOAT_VALUE
        self.epsilon = epsilon

    def update(self, value):
        self.max = max(self.max, value)
        self.min = min(self.min, value)

    def normalize(self, value):
        # DeepMind mctx logic: If stats are uninitialized (max < min),
        # pretend max = min = value. This ensures the numerator is 0.0
        # and natural evaluation forces the uninitialized Q-value to 0.0
        is_uninitialized = self.max < self.min
        maximum = value if is_uninitialized else self.max
        minimum = value if is_uninitialized else self.min

        diff = maximum - minimum
        epsilon = self.epsilon

        # Denominator calculation (works for both Tensors and standard floats)
        import torch

        if isinstance(value, torch.Tensor):
            denom = torch.clamp(torch.as_tensor(diff, device=value.device), min=epsilon)
            return ((value - minimum) / denom).clamp(0.0, 1.0)
        else:
            denom = max(diff, epsilon)
            return min(max((value - minimum) / denom, 0.0), 1.0)

    def __repr__(self):
        return f"min: {self.min}, max: {self.max}"
