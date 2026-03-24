from typing import Any, Dict, List, Optional, Tuple, Type
from .base import BaseExecutor
from agents.workers.payloads import WorkerPayload
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

    def _fetch_available_results(self) -> List[Any]:
        results = []
        for worker_cls, worker in self.workers:
            type_name = worker_cls.__name__

            if self.worker_signals.get(type_name, False):
                if hasattr(worker, "collect"):
                    num_steps = self.worker_signals.pop(f"{type_name}_num_steps", 1000)
                    data = worker.collect(num_steps)
                elif hasattr(worker, "evaluate"):
                    num_episodes = self.worker_signals.pop(
                        f"{type_name}_num_episodes", 1
                    )
                    data = worker.evaluate(num_episodes)
                elif hasattr(worker, "reanalyze"):
                    batch_size = self.worker_signals.pop(f"{type_name}_batch_size", 32)
                    data = worker.reanalyze(batch_size)
                elif hasattr(worker, "play_sequence"):
                    data = worker.play_sequence()
                else:
                    data = {}

                # Package synchronous result into payload for BaseExecutor
                results.append(WorkerPayload(worker_type=type_name, metrics=data))

                self.worker_signals[type_name] = False

        return results

    def update_parameters(
        self,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ):
        for _, w in self.workers:
            w.update_parameters(weights=weights, hyperparams=hyperparams)

    def request_work(self, worker_type: Type, **kwargs):
        """Signals the trigger event and stores arguments for the specified worker type."""
        type_name = worker_type.__name__
        self.worker_signals[type_name] = True
        for k, v in kwargs.items():
            self.worker_signals[f"{type_name}_{k}"] = v

    def stop(self):
        self.workers = []
