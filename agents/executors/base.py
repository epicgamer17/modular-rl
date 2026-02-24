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
    def _launch_workers(self, worker_cls: Type, args: Tuple, num_workers: int):
        """Internal method to start worker instances/processes."""
        pass

    @abstractmethod
    def _fetch_available_results(self) -> List[Any]:
        """Internal method to fetch results from workers immediately."""
        pass

    @abstractmethod
    def update_weights(self, state_dict: Dict[str, Any]):
        """Updates the weights of the workers."""
        pass

    def launch(self, worker_cls: Type, args: Tuple, num_workers: int):
        """Initializes and starts a group of workers. Appends to existing workers."""
        self._launch_workers(worker_cls, args, num_workers)

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
            while len(results) < min_samples:
                fetch_and_route()
                if len(results) >= min_samples:
                    break
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
            mcts_sps_values = []
            for res in results:
                # Only calculate stats for items that look like TransitionBatch or have expected keys
                is_transition_batch = hasattr(res, "rewards") or (
                    isinstance(res, dict)
                    and ("transitions" in res or "episode_stats" in res)
                )

                if not is_transition_batch:
                    continue

                total_transitions += len(res)
                if hasattr(res, "stats") and "mcts_sps" in res.stats:
                    mcts_sps_values.append(res.stats["mcts_sps"])
                elif (
                    isinstance(res, dict)
                    and "episode_stats" in res
                    and "mcts_sps" in res["episode_stats"]
                ):
                    # Handle TransitionBatch-like dicts
                    mcts_sps_values.append(res["episode_stats"]["mcts_sps"])

                if hasattr(res, "rewards") and hasattr(res, "stats"):
                    # For multiplayer: get player 0's final reward from stats
                    final_player_rewards = res.stats.get("final_player_rewards", None)
                    if final_player_rewards:
                        # This is the proper multi-agent reward dict
                        player_ids = list(final_player_rewards.keys())
                        if player_ids:
                            scores.append(final_player_rewards[player_ids[0]])
                        else:
                            scores.append(sum(res.rewards))
                    else:
                        # Single player: sum all rewards
                        scores.append(sum(res.rewards))
                    lengths.append(len(res))

            current_time = time.time()
            elapsed = current_time - self._last_stats_time
            self._last_stats_time = current_time

            if scores:
                stats["score"] = sum(scores) / len(scores)
                stats["episode_length"] = sum(lengths) / len(lengths)
                stats["num_episodes"] = len(results)

            if elapsed > 0:
                stats["actor_fps"] = total_transitions / elapsed
            if mcts_sps_values:
                stats["mcts_sps"] = sum(mcts_sps_values) / len(mcts_sps_values)

        return stats
