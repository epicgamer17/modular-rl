from typing import Any, Dict, List, Optional, Tuple, Type
from .base import BaseExecutor


class LocalExecutor(BaseExecutor):
    """
    Executor that runs workers in the main thread synchronously.
    """

    def _launch_workers(self, worker_cls: Type, args: Tuple, num_workers: int):
        self.workers = [worker_cls(*args, worker_id=i) for i in range(num_workers)]
        for worker in self.workers:
            worker.setup()

    def _fetch_available_results(self) -> List[Any]:
        # For local executor, we run one episode per worker synchronously
        # to simulate "fetching available results"
        results = []
        for worker in self.workers:
            results.append(worker.play_sequence())
        return results

    def update_weights(
        self, state_dict: Dict[str, Any], params: Optional[Dict[str, Any]] = None
    ):
        for worker in self.workers:
            worker.agent_network.load_state_dict(state_dict)
            worker.update_parameters(params)

    def stop(self):
        self.workers = []
