import torch
import torch.nn as nn
import numpy as np
import pytest
import time
from typing import Optional, Any
from executors.local_executor import LocalExecutor
from executors.torch_mp_executor import TorchMPExecutor
from replay_buffers.sequence import Sequence


class MockPolicy:
    def __init__(self):
        self.model = None  # Remove potential unpicklable/complex torch objects

    def reset(self, state):
        pass

    def compute_action(self, state, info):
        return 0

    def load_state_dict(self, state_dict):
        pass


class MockActor:
    def __init__(self, reward_val=1.0):
        self.reward_val = reward_val
        self.policy = MockPolicy()

    def setup(self):
        pass

    def run_episode(self, stats_tracker=None):
        seq = Sequence(num_players=1)
        seq.append(observation=np.zeros(1), info={}, terminated=False, truncated=False)
        # Mock a sequence with 5 steps
        for i in range(5):
            seq.append(
                observation=np.zeros(1),
                info={},
                terminated=(i == 4),
                truncated=False,
                action=0,
                reward=self.reward_val,
            )
        return seq


def test_local_executor_collect_data():
    executor = LocalExecutor()
    executor.launch(MockActor, (10.0,), num_workers=2)

    # Collect 4 samples (2 workers each run 1 episode per fetch)
    # LocalExecutor synchronous impl runs 1 episode per worker per _fetch_available_results
    data, stats = executor.collect_data(min_samples=2)

    assert len(data) >= 2
    assert "avg_episode_reward" in stats
    assert stats["avg_episode_reward"] == 50.0  # 5 steps * 10.0
    assert stats["avg_episode_length"] == 5
    executor.stop()


def test_torch_mp_executor_collect_data():
    executor = TorchMPExecutor()
    # Need to use a smaller number of workers for tests to avoid overhead/hangs in some envs
    executor.launch(MockActor, (2.0,), num_workers=2)

    # Wait for some data to be generated
    data, stats = executor.collect_data(min_samples=2)

    assert len(data) >= 2
    assert "avg_episode_reward" in stats
    assert stats["avg_episode_reward"] == 10.0  # 5 steps * 2.0
    assert stats["avg_episode_length"] == 5
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
