from typing import Any, Dict, List, Optional, Tuple, Type
from executors.base import BaseExecutor


class LocalExecutor(BaseExecutor):
    """
    Executor that runs workers in the main thread synchronously.
    """

    def _launch_workers(self, worker_cls: Type, args: Tuple, num_workers: int):
        self.workers = [worker_cls(*args) for _ in range(num_workers)]
        for worker in self.workers:
            if hasattr(worker, "setup"):
                worker.setup()

    def _fetch_available_results(self) -> List[Any]:
        # For local executor, we run one episode per worker synchronously
        # to simulate "fetching available results"
        results = []
        for worker in self.workers:
            if hasattr(worker, "run_episode"):
                results.append(worker.run_episode())
        return results

    def update_weights(self, state_dict: Dict[str, Any]):
        for worker in self.workers:
            # Assumes worker has a policy with a model
            if hasattr(worker, "policy") and hasattr(worker.policy, "model"):
                worker.policy.model.load_state_dict(state_dict)

    def stop(self):
        self.workers = []
