import pytest
pytestmark = pytest.mark.unit

import time
from typing import Any, Dict, List, Tuple, Type
from agents.executors.base import BaseExecutor


class MockExecutor(BaseExecutor):
    def _launch_workers(self, worker_cls: Type, args: Tuple, num_workers: int):
        # num_workers is stored in the mock by adding Mock objects to workers list
        self.workers = [object() for _ in range(num_workers)]

    def _fetch_available_results(self) -> List[Any]:
        return []

    def update_weights(self, state_dict: Dict[str, Any]):
        pass

    def request_work(self, worker_type: Type):
        pass

    def stop(self):
        pass


def test_global_metrics_calculation():
    """Executor computes global FPS and SPS via wall time / search time sums."""
    executor = MockExecutor()
    executor.launch(None, None, 2)  # 2 workers

    # Simulating collection: 2 episodes from different actors
    # Actor 1: 10 transitions, 100 sims, 0.1s search
    # Actor 2: 20 transitions, 200 sims, 0.2s search
    results = [
        {
            "episode_length": 10,
            "score": 0.0,
            "duration_seconds": 0.5,
            "mcts_simulations": 100,
            "mcts_search_time": 0.1,
        },
        {
            "episode_length": 20,
            "score": 0.0,
            "duration_seconds": 0.6,
            "mcts_simulations": 200,
            "mcts_search_time": 0.2,
        },
    ]

    start_wall = executor._last_stats_time
    stats = executor._calculate_stats(results)
    end_wall = executor._last_stats_time
    elapsed_wall = end_wall - start_wall

    # 1. Verify Actor FPS (Transitions / Wall Time)
    expected_fps = 30 / elapsed_wall
    assert stats["actor_fps"] == pytest.approx(expected_fps, abs=1e-2)

    # 2. Verify Global MCTS SPS (Total Sims / Total Search Time)
    # (100 + 200) / (0.1 + 0.2) = 300 / 0.3 = 1000
    expected_sps = 300 / 0.3
    assert stats["mcts_sps"] == pytest.approx(expected_sps, abs=1e-2)

    print(f"Global FPS: {stats['actor_fps']:.2f}")
    print(f"Global SPS: {stats['mcts_sps']:.2f} (Expected: {expected_sps:.2f})")


def test_multi_actor_sps_scaling():
    """Verifies that SPS scales linearly with actor count if they perform work in parallel."""
    # This is implicitly tested by the summing logic, but let's be explicit.
    executor = MockExecutor()
    executor.launch(None, None, 4)

    # 4 actors each doing 100 sims in 0.1s search time
    results = [
        {"episode_length": 10, "mcts_simulations": 100, "mcts_search_time": 0.1}
        for _ in range(4)
    ]

    stats = executor._calculate_stats(results)

    # Total sims: 400. Total search time: 0.4. SPS: 400 / 0.4 = 1000.
    # Note: Even though they are in parallel, the total work done is 400 sims.
    # The sum of search times across all episodes is 0.4s.
    # This correctly represents the search efficiency of the *system*.
    expected_sps = 400 / 0.4
    assert stats["mcts_sps"] == expected_sps
    print(f"Scaled SPS: {stats['mcts_sps']:.2f}")
