import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type


class BaseExecutor(ABC):
    """
    Base class for executors that manage data collection workers.
    """

    def __init__(self):
        self.workers = []
        # Buffer for results from different worker types
        # Stores {worker_type_name: [results]}
        self.result_buffer = {}

    @abstractmethod
    def _launch_workers(self, worker_cls: Type, args: Tuple, num_workers: int, **kwargs):
        """Internal method to start worker instances/processes."""
        pass  # pragma: no cover

    @abstractmethod
    def _fetch_available_results(self) -> List[Any]:
        """Internal method to fetch results from workers immediately."""
        pass  # pragma: no cover

    @abstractmethod
    def update_weights(
        self, state_dict: Dict[str, Any], params: Optional[Dict[str, Any]] = None
    ):
        """Updates the weights of the workers."""
        pass  # pragma: no cover

    @abstractmethod
    def request_work(self, worker_type: Type):
        """
        Requests that workers of a specific type perform their task.
        For some executors, this signals an event to wake up idle workers.
        """
        pass  # pragma: no cover

    def launch(self, worker_cls: Type, args: Tuple, num_workers: int, **kwargs):
        """Initializes and starts a group of workers. Appends to existing workers."""
        self._launch_workers(worker_cls, args, num_workers, **kwargs)

    def collect_data(
        self, min_samples: Optional[int] = None, worker_type: Optional[Type] = None
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Collects data from workers, accumulating until min_samples is reached.
        If min_samples is None, returns whatever is currently available.

        Args:
            min_samples: Minimum number of samples to collect before returning.
            worker_type: If provided, only returns results from this worker type.
                         Other types are buffered internally.

        Returns:
            Tuple of (list of data items, dict of statistics).
        """
        results = []
        type_name = worker_type.__name__ if worker_type else None

        # 1. Take from buffer first if type specified
        if type_name and type_name in self.result_buffer:
            results.extend(self.result_buffer.pop(type_name))

        # 2. Fetch new results and route them
        def fetch_and_route():
            new_batch = self._fetch_available_results()
            for obj in new_batch:
                # Results should be tuples/dicts/objects with type info if mixed
                # For backward compatibility, if it's not a (type, data) tuple,
                # we assume it matches the current request or default.
                if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], str):
                    t_name, data = obj
                    if type_name and t_name == type_name:
                        results.append(data)
                    else:
                        if t_name not in self.result_buffer:
                            self.result_buffer[t_name] = []
                        self.result_buffer[t_name].append(data)
                else:
                    # Default: assume it's what we asked for
                    results.append(obj)

        if min_samples is not None:
            while True:
                current_samples = sum(
                    r.get("num_samples", 0) for r in results if isinstance(r, dict)
                )
                if current_samples >= min_samples:
                    break

                old_len = len(results)
                fetch_and_route()
                if len(results) == old_len:
                    # No new results being returned, break to avoid infinite loop
                    break

                # Sleep more if no progress made to avoid busy waiting in multi-process case
                time.sleep(0.01)
        else:
            fetch_and_route()

        # Stats and telemetry are now handled by TelemetryComponent inside the Engine/Worker
        # The executor simply yields whatever the results contain.
        return results, {}
