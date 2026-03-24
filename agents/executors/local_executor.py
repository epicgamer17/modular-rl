from typing import Any, Dict, List, Optional, Tuple, Type
from .base import BaseExecutor
from agents.workers.payloads import WorkerPayload, TaskRequest, TaskType
import torch


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
        # Type-name to task mapping
        self.pending_tasks = {}

    def _fetch_available_results(self) -> List[Any]:
        results = []
        for worker_cls, worker in self.workers:
            type_name = worker_cls.__name__

            task = self.pending_tasks.pop(type_name, None)
            if task:
                payload = worker.execute(task)
                results.append(payload)

        return results

    def update_parameters(
        self,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ):
        for _, w in self.workers:
            w.update_parameters(weights=weights, hyperparams=hyperparams)

    def request_work(self, worker_type: Type, **kwargs):
        """Signals that work is pending and stores the TaskRequest for the specified worker type."""
        type_name = worker_type.__name__
        
        # Map legacy kwargs to TaskType
        task_type = None
        batch_size = 0
        
        if "num_steps" in kwargs:
            task_type = TaskType.COLLECT
            batch_size = kwargs.pop("num_steps")
        elif "num_episodes" in kwargs:
            task_type = TaskType.EVALUATE
            batch_size = kwargs.pop("num_episodes")
        elif "batch_size" in kwargs:
            task_type = TaskType.REANALYZE
            batch_size = kwargs.pop("batch_size")
        else:
            # Default to collect if nothing specified
            task_type = TaskType.COLLECT
            batch_size = 1000
            
        request = TaskRequest(task_type=task_type, batch_size=batch_size, kwargs=kwargs)
        self.pending_tasks[type_name] = request

    def stop(self):
        self.workers = []
