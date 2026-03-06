import pytest
pytestmark = pytest.mark.unit

from agents.executors.local_executor import LocalExecutor


class MockWorker:
    def __init__(self, *args, **kwargs):
        self.worker_id = kwargs.get("worker_id", 0)
        self.result = args[0] if args else "default"

    def setup(self):
        pass

    def play_sequence(self):
        return {"episode_length": 1, "score": 0.0, "payload": self.result}


class MockTester:
    def __init__(self, *args, **kwargs):
        self.worker_id = kwargs.get("worker_id", 0)
        self.result = args[0] if args else "test_result"

    def setup(self):
        pass

    def play_sequence(self):
        return {"episode_length": 1, "score": 0.0, "payload": self.result}


def test_routing_and_buffering():
    executor = LocalExecutor()

    # 1. Launch primary workers
    executor.launch(MockWorker, ("actor_data",), num_workers=2)

    # 2. Launch tester worker
    executor.launch(MockTester, ("tester_data",), num_workers=1)

    # Total workers should be 3
    assert len(executor.workers) == 3

    # 3. Collect only Actor data
    actor_results, _ = executor.collect_data(min_samples=1, worker_type=MockWorker)

    # Should have 2 actor results (from the fetch + routing)
    assert len(actor_results) == 2
    assert all(r["payload"] == "actor_data" for r in actor_results)

    # Tester result should be in buffer
    assert "MockTester" in executor.result_buffer
    assert [r["payload"] for r in executor.result_buffer["MockTester"]] == [
        "tester_data"
    ]

    # 4. Collect Tester data
    tester_results, _ = executor.collect_data(min_samples=1, worker_type=MockTester)
    # Should get 1 from buffer. Since min_samples=1 is satisfied, it won't fetch more.
    assert len(tester_results) == 1
    assert [r["payload"] for r in tester_results] == ["tester_data"]

    # Buffer should now be empty for Tester
    assert "MockTester" not in executor.result_buffer


def test_buffer_persistence():
    executor = LocalExecutor()
    executor.launch(MockTester, ("t1",), num_workers=1)

    # Collect actor data (none exists), this will fetch everything and buffer the tester result
    actor_data, _ = executor.collect_data(min_samples=None, worker_type=MockWorker)
    assert len(actor_data) == 0
    assert [r["payload"] for r in executor.result_buffer["MockTester"]] == ["t1"]

    # Launch another tester
    executor.launch(MockTester, ("t2",), num_workers=1)

    # Collect tester data
    # Should get t1 from buffer (len=1). min_samples=2 is not satisfied (1 < 2).
    # Should then fetch from ALL workers (both testers).
    # Result: t1 (buffer) + t1 (worker 0) + t2 (worker 1) = 3 results
    tester_data, _ = executor.collect_data(min_samples=2, worker_type=MockTester)
    assert len(tester_data) == 3
    assert [r["payload"] for r in tester_data].count("t1") == 2
    assert [r["payload"] for r in tester_data].count("t2") == 1
