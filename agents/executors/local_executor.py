from typing import Any, Dict, List, Optional, Tuple, Type
from .base import BaseExecutor


class LocalExecutor(BaseExecutor):
    """
    Executor that runs workers in the main thread synchronously.
    """

    def __init__(self):
        super().__init__()
        self.worker_signals = {}

    def _launch_workers(self, worker_cls: Type, args: Tuple, num_workers: int):
        new_workers = [
            (worker_cls, worker_cls(*args, worker_id=i)) for i in range(num_workers)
        ]
        for _, worker in new_workers:
            worker.setup()
        self.workers.extend(new_workers)

    def _fetch_available_results(self) -> List[Any]:
        # For local executor, we run one episode per worker synchronously
        # to simulate "fetching available results"
        results = []
        for worker_cls, worker in self.workers:
            type_name = worker_cls.__name__

            # Use signaling for Tester to prevent running it continuously
            use_signaling = type_name == "Tester"

            if use_signaling:
                if self.worker_signals.get(type_name, False):
                    results.append((type_name, worker.play_sequence()))
                    # Reset tester signal after running
                    self.worker_signals[type_name] = False
            else:
                results.append((type_name, worker.play_sequence()))

        return results

    def update_weights(
        self, state_dict: Dict[str, Any], params: Optional[Dict[str, Any]] = None
    ):
        for _, w in self.workers:
            w.agent_network.load_state_dict(state_dict)
            if params is not None and hasattr(w, "action_selector"):
                w.action_selector.update_parameters(params)
            w.update_parameters(params)

    def request_work(self, worker_type: Type):
        """Signals the trigger event for the specified worker type."""
        type_name = worker_type.__name__
        self.worker_signals[type_name] = True

    def stop(self):
        self.workers = []
