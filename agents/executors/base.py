import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type
from agents.workers.payloads import WorkerPayload
import torch


class BaseExecutor(ABC):
    """
    Base class for executors that manage data collection workers.
    """

    def __init__(self):
        self.workers = []
        self._last_stats_time = time.time()
        # Buffer for results from different worker types
        # Stores {worker_type_name: [results]}
        self.result_buffer = {}

    @abstractmethod
    def _launch_workers(self, worker_cls: Type, args: Tuple, num_workers: int):
        """Internal method to start worker instances/processes."""
        pass  # pragma: no cover

    @abstractmethod
    def _fetch_available_results(self) -> List[Any]:
        """Internal method to fetch results from workers immediately."""
        pass  # pragma: no cover

    @abstractmethod
    def update_parameters(
        self,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ):
        """Updates the weights and/or hyperparameters of the workers."""
        pass  # pragma: no cover

    @abstractmethod
    def request_work(self, worker_type: Type, **kwargs):
        """
        Requests that workers of a specific type perform their task.
        Args:
            worker_type: Class reference for the target workers.
            **kwargs: Arguments for the worker task (e.g. num_steps, num_episodes).
        """
        pass  # pragma: no cover

    def launch_workers(self, worker_cls: Type, args: Tuple, num_workers: int):
        """Initializes and starts a group of workers. Appends to existing workers."""
        self._launch_workers(worker_cls, args, num_workers)

    def collect_data(
        self, num_steps: Optional[int] = None, worker_type: Optional[Type] = None
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Collects data from workers.
        If num_steps is provided, it explicitly triggers workers to collect those steps.

        Args:
            num_steps: If provided, requests workers to collect this many steps.
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

        # 2. Trigger work if num_steps requested
        if num_steps is not None and worker_type:
            self.request_work(worker_type, num_steps=num_steps)

        # 3. Fetch new results and route them
        def fetch_and_route():
            new_batch = self._fetch_available_results()
            for obj in new_batch:
                if isinstance(obj, WorkerPayload):
                    t_name = obj.worker_type
                    # Standard interface: we primarily return metrics to the trainer
                    # but ensure trajectory data is included if it exists in the payload.
                    data = obj.metrics
                    if obj.data is not None:
                        if not data:
                            data = obj.data
                        elif isinstance(obj.data, dict):
                            data = {**data, **obj.data}

                    if type_name and t_name == type_name:
                        results.append(data)
                    else:
                        if t_name not in self.result_buffer:
                            self.result_buffer[t_name] = []
                        self.result_buffer[t_name].append(data)
                elif isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], str):
                    # Legacy support for raw tuples
                    t_name, data = obj
                    if type_name and t_name == type_name:
                        results.append(data)
                    else:
                        if t_name not in self.result_buffer:
                            self.result_buffer[t_name] = []
                        self.result_buffer[t_name].append(data)
                else:
                    results.append(obj)

        if num_steps is not None and worker_type:
            # For synchronous requests, we wait until we have at least one result or a timeout
            # (In LocalExecutor this is immediate, in TorchMP it might take a moment)
            while not results:
                fetch_and_route()
                if not results:
                    time.sleep(0.001)
        else:
            fetch_and_route()

        # Stats calculation logic (only for TransitionBatch-like objects)
        stats = self._calculate_stats(results)

        return results, stats

    def _calculate_stats(self, results: List[Any]) -> Dict[str, Any]:
        """Calculates statistics from the collected data items."""
        stats = {}
        if results:
            scores = []
            lengths = []
            total_transitions = 0
            total_duration = 0.0
            total_mcts_simulations = 0
            total_mcts_search_time = 0.0

            for res in results:
                # If res is a dictionary representing episode_stats from BaseActor
                if isinstance(res, dict) and "episode_length" in res:
                    total_transitions += res["episode_length"]
                    total_duration += res.get("duration_seconds", 0.0)

                    if "mcts_simulations" in res:
                        total_mcts_simulations += res["mcts_simulations"]
                    if "mcts_search_time" in res:
                        total_mcts_search_time += res["mcts_search_time"]

                    if "final_player_rewards" in res:
                        player_ids = list(res["final_player_rewards"].keys())
                        if player_ids:
                            scores.append(res["final_player_rewards"][player_ids[0]])
                        else:
                            scores.append(res.get("score", 0.0))
                    else:
                        scores.append(res.get("score", 0.0))

                    if "final_episode_length" in res:
                        lengths.append(res["final_episode_length"])
                    else:
                        lengths.append(res["episode_length"])
                # We skip Tester results here, as they are processed elsewhere (e.g. Trainers' process_test_results)

            current_time = time.time()
            elapsed_wall = current_time - self._last_stats_time
            self._last_stats_time = current_time

            if scores:
                stats["score"] = sum(scores) / len(scores)
                stats["episode_length"] = sum(lengths) / len(lengths)
                stats["num_episodes"] = len(results)

            if elapsed_wall > 0 and total_transitions > 0:
                stats["actor_fps"] = total_transitions / elapsed_wall

            if total_mcts_search_time > 0:
                stats["mcts_sps"] = total_mcts_simulations / total_mcts_search_time

        return stats
