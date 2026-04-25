"""Stateful services backing the MetricsSink operator.

`MetricsTracker` computes rolling throughput (SPS/UPS) from monotonic step
counters. `MetricsStore` owns the process-wide default tracker so the
operator stays stateless.
"""

from typing import Dict, Optional
import time


class MetricsTracker:
    """Tracks throughput (SPS/UPS) from actor/learner step counters."""

    def __init__(self):
        self.start_time = time.time()
        self.last_actor_step = 0
        self.last_learner_step = 0
        self.last_time = self.start_time
        self.last_sps = 0.0
        self.last_ups = 0.0

    def update(
        self, current_actor_step: int, current_learner_step: int
    ) -> Dict[str, float]:
        now = time.time()
        elapsed = now - self.last_time
        if elapsed < 0.001:
            return {"sps": self.last_sps, "ups": self.last_ups}

        actor_delta = current_actor_step - self.last_actor_step
        learner_delta = current_learner_step - self.last_learner_step

        if actor_delta > 0:
            self.last_sps = actor_delta / elapsed

        if learner_delta > 0:
            self.last_ups = learner_delta / elapsed

        self.last_time = now
        self.last_actor_step = current_actor_step
        self.last_learner_step = current_learner_step

        return {"sps": self.last_sps, "ups": self.last_ups}


class MetricsStore:
    """Process-wide holder for the default MetricsTracker."""

    _default: Optional[MetricsTracker] = None

    @classmethod
    def default(cls) -> MetricsTracker:
        if cls._default is None:
            cls._default = MetricsTracker()
        return cls._default

    @classmethod
    def reset(cls) -> None:
        cls._default = None
