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

    def normalize(self, value: float) -> float:
        if self.max > self.min:
            # We normalize only when we have at a max and min value
            denom = self.max - self.min
            if self.soft_update:
                denom = max(denom, self.min_max_epsilon)
            return (value - self.min) / denom
        return value

    def __repr__(self):
        return f"min: {self.min}, max: {self.max}"
