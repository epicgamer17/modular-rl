import pytest

pytestmark = pytest.mark.integration

import torch
import torch.nn as nn
import time
from typing import Optional, Any
from agents.executors.local_executor import LocalExecutor
from agents.executors.torch_mp_executor import TorchMPExecutor
from replay_buffers.sequence import Sequence


class MockPolicy:
    def __init__(self):
        self.agent_network = None  # Remove potential unpicklable/complex torch objects

    def reset(self, state):
        pass

    def compute_action(self, state, info):
        return 0

    def load_state_dict(self, state_dict):
        pass


class MockActor:
    def __init__(self, *args, worker_id=0):
        self.reward_val = args[0] if args else 1.0
        self.worker_id = worker_id
        self.policy = MockPolicy()

    def setup(self):
        pass

    def play_sequence(self, stats_tracker=None):
        return {
            "episode_length": 5,
            "score": 5.0 * self.reward_val,
            "duration_seconds": 0.1,
        }


def test_local_executor_collect_data():
    executor = LocalExecutor()
    executor.launch(MockActor, (10.0,), num_workers=2)

    # Collect 4 samples (2 workers each run 1 episode per fetch)
    # LocalExecutor synchronous impl runs 1 episode per worker per _fetch_available_results
    data, stats = executor.collect_data(min_samples=2)

    assert len(data) >= 2
    assert "score" in stats
    assert stats["score"] == 10.0  # 5 steps * 2.0 (wait, worker_args is (10.0,) above?)
    assert stats["episode_length"] == 5
    executor.stop()


def test_torch_mp_executor_collect_data():
    executor = TorchMPExecutor()
    # Need to use a smaller number of workers for tests to avoid overhead/hangs in some envs
    executor.launch(MockActor, (2.0,), num_workers=2)

    # Wait for some data to be generated
    data, stats = executor.collect_data(min_samples=2)

    assert len(data) >= 2
    assert "score" in stats
    assert stats["score"] == 10.0  # 5 steps * 2.0
    assert stats["episode_length"] == 5
    executor.stop()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    try:
        print("Running local executor test...", flush=True)
        test_local_executor_collect_data()
        print("Local executor test passed!", flush=True)

        print("Running torch mp executor test...", flush=True)
        test_torch_mp_executor_collect_data()
        print("Torch mp executor test passed!", flush=True)

        print("All tests passed!", flush=True)
    except Exception as e:
        print(f"Test failed: {e}", flush=True)
        import traceback

        traceback.print_exc()
        exit(1)
