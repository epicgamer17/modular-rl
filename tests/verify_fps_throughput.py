import time
import unittest
from typing import Any, Dict, List, Tuple, Type
from agents.executors.base import BaseExecutor


class MockSequence:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


class MockExecutor(BaseExecutor):
    def _launch_workers(self, worker_cls: Type, args: Tuple, num_workers: int):
        pass

    def _fetch_available_results(self) -> List[Any]:
        return []

    def update_weights(self, state_dict: Dict[str, Any]):
        pass

    def stop(self):
        pass


class TestFPSThroughput(unittest.TestCase):
    def test_throughput_calculation(self):
        executor = MockExecutor()

        # Simulating first collection
        time.sleep(0.1)
        results = [MockSequence(10), MockSequence(20)]  # 30 transitions
        start_time = executor._last_stats_time
        stats = executor._calculate_stats(results)
        end_time = executor._last_stats_time

        elapsed = end_time - start_time
        expected_fps = 30 / elapsed

        self.assertAlmostEqual(stats["actor_fps"], expected_fps, places=2)
        print(f"Test 1 FPS: {stats['actor_fps']:.2f} (Expected: {expected_fps:.2f})")

        # Simulating second collection after some time
        time.sleep(0.2)
        results = [MockSequence(50)]  # 50 transitions
        start_time = executor._last_stats_time
        stats = executor._calculate_stats(results)
        end_time = executor._last_stats_time

        elapsed = end_time - start_time
        expected_fps = 50 / elapsed

        self.assertAlmostEqual(stats["actor_fps"], expected_fps, places=2)
        print(f"Test 2 FPS: {stats['actor_fps']:.2f} (Expected: {expected_fps:.2f})")


if __name__ == "__main__":
    unittest.main()
