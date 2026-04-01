import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type


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
                current_transitions = sum(
                    r.get("episode_length", 0) for r in results if isinstance(r, dict)
                )
                if current_transitions >= min_samples:
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
