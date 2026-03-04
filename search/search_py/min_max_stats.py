import collections
from typing import Optional, List

MAXIMUM_FLOAT_VALUE = float("inf")

# TODO: EFFICIENT ZERO SOFT MINMAX STATS


class MinMaxStats(object):
    def __init__(
        self,
        known_bounds: Optional[List[float]],
        soft_update: bool = False,
        min_max_epsilon: float = 0.01,
    ):
        self.soft_update = soft_update
        self.min_max_epsilon = min_max_epsilon
        self.max = known_bounds[1] if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.min = known_bounds[0] if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.max = max(self.max, value)
        self.min = min(self.min, value)

    def normalize(self, value):
        epsilon = self.min_max_epsilon if self.soft_update else 1e-8

        # DeepMind mctx logic: If stats are uninitialized (max < min),
        # pretend max = min = value. This ensures the numerator is 0.0
        # and natural evaluation forces the uninitialized Q-value to 0.0
        # rather than returning un-normalized values or crashing.
        is_uninitialized = self.max < self.min
        maximum = value if is_uninitialized else self.max
        minimum = value if is_uninitialized else self.min

        diff = maximum - minimum

        # Denominator calculation (works for both Tensors and standard floats)
        import torch

        if isinstance(diff, torch.Tensor):
            denom = torch.clamp(diff, min=epsilon)
        else:
            denom = max(diff, epsilon)

        return (value - minimum) / denom

    def __repr__(self):
        return f"min: {self.min}, max: {self.max}"
