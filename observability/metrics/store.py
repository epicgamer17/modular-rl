from typing import Dict, Any, List, Optional, Union
import time
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MetricPoint:
    """A single data point for a metric."""

    value: float
    timestamp: float = field(default_factory=time.time)
    step: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricStore:
    """
    Unified abstraction for all execution metrics.
    Supports rolling windows, EMA tracking, and series analysis.
    """

    def __init__(self):
        self._data: Dict[str, List[MetricPoint]] = {}
        self._emas: Dict[str, float] = {}
        self._ema_alpha = 0.05  # Default EMA alpha

        # Performance tracking (legacy/compatibility)
        self._start_time = time.time()
        self._last_actor_step = 0
        self._last_learner_step = 0
        self._last_time = self._start_time
        self._last_sps = 0.0
        self._last_ups = 0.0

    def log(
        self,
        name: str,
        value: float,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log a metric value at a specific step."""
        if name not in self._data:
            self._data[name] = []

        point = MetricPoint(value=float(value), step=step, metadata=metadata or {})
        self._data[name].append(point)

        # Update EMA tracking
        if name not in self._emas:
            self._emas[name] = float(value)
        else:
            self._emas[name] = (
                self._ema_alpha * float(value)
                + (1 - self._ema_alpha) * self._emas[name]
            )

    def get(self, name: str) -> List[MetricPoint]:
        """Retrieve all points for a given metric."""
        return self._data.get(name, [])

    def series(self, name: str) -> List[float]:
        """Get the raw value series for a metric."""
        return [p.value for p in self.get(name)]

    def get_ema(self, name: str) -> Optional[float]:
        """Get the current EMA value for a metric."""
        return self._emas.get(name)

    def rolling_average(self, name: str, window: int = 10) -> List[float]:
        """Compute rolling average series."""
        values = self.series(name)
        if len(values) < window:
            return values

        weights = np.ones(window) / window
        return np.convolve(values, weights, mode="valid").tolist()

    def compute_rates(
        self, current_actor_step: int, current_learner_step: int
    ) -> Dict[str, float]:
        """Compute throughput rates (SPS, UPS)."""
        now = time.time()
        elapsed = now - self._last_time

        if elapsed < 0.001:
            return {"sps": self._last_sps, "ups": self._last_ups}

        actor_delta = current_actor_step - self._last_actor_step
        learner_delta = current_learner_step - self._last_learner_step

        if actor_delta > 0:
            self._last_sps = actor_delta / elapsed
        if learner_delta > 0:
            self._last_ups = learner_delta / elapsed

        self._last_time = now
        self._last_actor_step = current_actor_step
        self._last_learner_step = current_learner_step

        return {"sps": self._last_sps, "ups": self._last_ups}

    def sync(self):
        """Placeholder for future distributed synchronization."""
        pass


# Global instance for easy access
_GLOBAL_STORE = MetricStore()


def get_global_store() -> MetricStore:
    return _GLOBAL_STORE
