import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type


class BaseExecutor(ABC):
    """
    Base class for executors that manage data collection workers.
    """

    def __init__(self):
        self.workers = []

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
            fps_values = []
            for res in results:
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

                    # FPS tracking from game duration
                    if hasattr(res, "duration_seconds") and res.duration_seconds > 0:
                        fps_values.append(len(res) / res.duration_seconds)

            if scores:
                stats["score"] = sum(scores) / len(scores)
                stats["episode_length"] = sum(lengths) / len(lengths)
                stats["num_episodes"] = len(results)
            if fps_values:
                stats["actor_fps"] = sum(fps_values) / len(fps_values)

        return stats
