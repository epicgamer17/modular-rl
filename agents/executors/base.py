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

    @abstractmethod
    def stop(self):
        """Stops all workers."""
        pass

    def launch(self, worker_cls: Type, args: Tuple, num_workers: int):
        """Initializes and starts the workers."""
        self._launch_workers(worker_cls, args, num_workers)

    def collect_data(
        self, min_samples: Optional[int] = None
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Collects data from workers, accumulating until min_samples is reached.
        If min_samples is None, returns whatever is currently available.

        Returns:
            Tuple of (list of data items, dict of statistics).
        """
        results = []

        if min_samples is not None:
            while len(results) < min_samples:
                new_results = self._fetch_available_results()
                if new_results:
                    results.extend(new_results)
                else:
                    time.sleep(0.01)  # Minimal backoff
        else:
            results.extend(self._fetch_available_results())

        # Stats calculation logic
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
