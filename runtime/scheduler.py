"""
Scheduler and Loop nodes for the RL IR runtime.
Handles periodic tasks, main execution loops, and parallel rollout pooling.
"""

from typing import List, Callable, Any, Optional, Dict
import threading
import time
from runtime.runtime import ActorRuntime, LearnerRuntime
from dataclasses import dataclass, field


@dataclass
class SchedulePlan:
    """
    Declarative execution plan for RL loops.
    """

    actor_frequency: int = 1
    learner_frequency: int = 1
    prefetch_depth: int = 1
    batching_strategy: str = "serial"  # "serial", "parallel", "vectorized"
    sync_points: List[str] = field(default_factory=list)  # ["step", "episode", "epoch"]

    def to_dict(self):
        return {
            "actor_frequency": self.actor_frequency,
            "learner_frequency": self.learner_frequency,
            "prefetch_depth": self.prefetch_depth,
            "batching_strategy": self.batching_strategy,
            "sync_points": self.sync_points,
        }


from typing import List, Callable, Any, Optional, Dict, Union


class ScheduleExecutor:
    """
    Executes a SchedulePlan by coordinating Actor and Learner runtimes.
    Supports parallel and serial execution strategies.
    """

    def __init__(
        self,
        plan: SchedulePlan,
        actor_runtime: Union[ActorRuntime, List[ActorRuntime]],
        learner_runtime: LearnerRuntime,
    ):
        self.plan = plan
        self.actor_runtimes = (
            actor_runtime if isinstance(actor_runtime, list) else [actor_runtime]
        )
        self.learner_runtime = learner_runtime
        self._running = False

    def run(self, total_actor_steps: int):
        self._running = True
        actor_steps = 0

        while actor_steps < total_actor_steps and self._running:
            # 1. Actor Execution (Strategy-aware)
            current_steps = self._execute_actors()
            actor_steps += current_steps

            # 2. Learner Step(s)
            for _ in range(self.plan.learner_frequency):
                # Learner usually needs data from a buffer
                # In this simplified model, we assume update_step handles its own sampling
                self.learner_runtime.update_step()

            if actor_steps >= total_actor_steps:
                break

    def _execute_actors(self) -> int:
        """Executes actor steps based on the batching strategy."""
        num_steps = 0

        if self.plan.batching_strategy == "parallel" and len(self.actor_runtimes) > 1:
            # Multi-threaded parallel execution
            threads = []
            for runtime in self.actor_runtimes:
                for _ in range(self.plan.actor_frequency):
                    t = threading.Thread(target=runtime.step)
                    threads.append(t)
                    t.start()
                    num_steps += 1
            for t in threads:
                t.join()
        else:
            # Serial execution (standard or single actor)
            for runtime in self.actor_runtimes:
                for _ in range(self.plan.actor_frequency):
                    runtime.step()
                    num_steps += 1

        return num_steps

    def stop(self):
        self._running = False
