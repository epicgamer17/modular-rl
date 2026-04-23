"""
Scheduler and Loop nodes for the RL IR runtime.
Handles periodic tasks, main execution loops, and parallel rollout pooling.
"""

import threading
from typing import List, Callable, Any, Optional, Dict
import time
from runtime.runtime import ActorRuntime

class EveryN:
    """Triggers an action every N steps."""
    def __init__(self, n: int, action: Callable[[], Any]):
        self.n = n
        self.action = action
        self._count = 0

    def step(self) -> Optional[Any]:
        self._count += 1
        if self._count >= self.n:
            result = self.action()
            self._count = 0
            return result
        return None

class ParallelActorPool:
    """
    Executes multiple ActorRuntimes in parallel using threading.
    """
    def __init__(self, runtimes: List[ActorRuntime]):
        self.runtimes = runtimes

    def rollout(self, steps_per_actor: int) -> List[List[Dict[str, Any]]]:
        """
        Runs all actors in parallel for a fixed number of steps.
        """
        results = [None] * len(self.runtimes)
        threads = []

        def worker(idx, runtime, steps):
            results[idx] = runtime.collect_trajectory(steps)

        for i, runtime in enumerate(self.runtimes):
            t = threading.Thread(target=worker, args=(i, runtime, steps_per_actor))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results

class Loop:
    """
    A high-level execution loop coordinating interaction and training.
    """
    def __init__(
        self,
        interact_fn: Callable[[], Any],
        train_fn: Callable[[], Any],
        every_n_train: int = 1
    ):
        self.interact_fn = interact_fn
        self.train_fn = train_fn
        self.trainer = EveryN(every_n_train, train_fn)
        self.running = False

    def run(self, total_steps: int):
        self.running = True
        for _ in range(total_steps):
            if not self.running:
                break
            self.interact_fn()
            self.trainer.step()

    def stop(self):
        self.running = False
