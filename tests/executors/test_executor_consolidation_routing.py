import pytest
pytestmark = pytest.mark.integration

import torch
import torch.multiprocessing as mp
import time
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
        return {"episode_length": 1, "score": 0.0, "payload": "actor_data"}


class MockTesterWorker:
    def __init__(self, *args, **kwargs):
        self.worker_id = kwargs.get("worker_id", 0)

    def setup(self):
        pass

    def play_sequence(self):
        return {"episode_length": 1, "score": 0.0, "payload": "tester_data"}


def test_local_executor_consolidation():
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
    actor_data, _ = executor.collect_data(min_samples=1, worker_type=MockActorWorker)
    assert len(actor_data) == 2
    assert all(r["payload"] == "actor_data" for r in actor_data)

    # 4. Collect Tester data
    tester_data, _ = executor.collect_data(min_samples=1, worker_type=MockTesterWorker)
    assert [r["payload"] for r in tester_data] == ["tester_data"]


def test_torch_mp_executor_consolidation():
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
    actor_data, _ = executor.collect_data(min_samples=None, worker_type=MockActorWorker)
    assert len(actor_data) > 0
    assert all(d["payload"] == "actor_data" for d in actor_data)

    # 4. Collect Tester data
    tester_data, _ = executor.collect_data(
        min_samples=None, worker_type=MockTesterWorker
    )
    assert len(tester_data) > 0
    assert all(d["payload"] == "tester_data" for d in tester_data)

    executor.stop()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
