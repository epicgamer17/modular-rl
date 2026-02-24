import torch
import torch.multiprocessing as mp
import time
import unittest
from typing import Any, Tuple, Type

from agents.executors.local_executor import LocalExecutor
from agents.executors.torch_mp_executor import TorchMPExecutor


def dummy_env_factory():
    return None


class MockActorWorker:
    def __init__(self, *args, **kwargs):
        self.worker_id = kwargs.get("worker_id", 0)

    def setup(self):
        pass

    def play_sequence(self):
        return "actor_data"


class MockTesterWorker:
    def __init__(self, *args, **kwargs):
        self.worker_id = kwargs.get("worker_id", 0)

    def setup(self):
        pass

    def play_sequence(self):
        return "tester_data"


class TestExecutorConsolidation(unittest.TestCase):
    def test_local_executor_consolidation(self):
        executor = LocalExecutor()

        # 1. Launch Actor
        actor_args = (None, None, None, 1, {}, torch.device("cpu"), "actor")
        executor.launch(MockActorWorker, actor_args, num_workers=2)

        # 2. Launch Tester
        tester_args = (
            dummy_env_factory,
            None,
            None,
            1,
            {},
            torch.device("cpu"),
            "tester",
            [],
        )
        executor.launch(MockTesterWorker, tester_args, num_workers=1)

        # 3. Collect only Actor data
        actor_data, _ = executor.collect_data(
            min_samples=1, worker_type=MockActorWorker
        )
        self.assertEqual(len(actor_data), 2)
        self.assertTrue(all(r == "actor_data" for r in actor_data))

        # 4. Collect Tester data
        tester_data, _ = executor.collect_data(
            min_samples=1, worker_type=MockTesterWorker
        )
        self.assertEqual(tester_data, ["tester_data"])

    def test_torch_mp_executor_consolidation(self):
        executor = TorchMPExecutor()

        # 1. Launch Actor
        actor_args = (None, None, None, 1, {}, torch.device("cpu"), "actor")
        executor.launch(MockActorWorker, actor_args, num_workers=1)

        # 2. Launch Tester
        tester_args = (
            dummy_env_factory,
            None,
            None,
            1,
            {},
            torch.device("cpu"),
            "tester",
            [],
        )
        executor.launch(MockTesterWorker, tester_args, num_workers=1)

        # Give some time for workers to generate data
        time.sleep(1.0)

        # 3. Collect Actor data
        actor_data, _ = executor.collect_data(
            min_samples=None, worker_type=MockActorWorker
        )
        self.assertTrue(len(actor_data) > 0)
        self.assertTrue(all(d == "actor_data" for d in actor_data))

        # 4. Collect Tester data
        tester_data, _ = executor.collect_data(
            min_samples=None, worker_type=MockTesterWorker
        )
        self.assertTrue(len(tester_data) > 0)
        self.assertTrue(all(d == "tester_data" for d in tester_data))

        executor.stop()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    unittest.main()
