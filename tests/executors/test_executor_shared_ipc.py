import pytest

pytestmark = pytest.mark.integration

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Optional, Any, Tuple
from agents.executors.torch_mp_executor import TorchMPExecutor
from replay_buffers.sequence import Sequence


class MockNetwork(nn.Module):
    def __init__(self, input_shape=(4,), num_actions=2):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.observation_dtype = torch.float32


class MockGame:
    def __init__(self):
        self.num_actions = 2
        self.num_players = 1

    def env_factory(self):
        return None


class MockConfig:
    def __init__(self):
        self.max_episode_length = 10
        self.multi_process = True
        self.num_workers = 2
        self.game = MockGame()


class MockActor:
    def __init__(
        self,
        env_factory,
        agent_network,
        selector,
        num_players,
        config,
        device,
        name,
        worker_id=0,
    ):
        self.worker_id = worker_id
        self.num_players = num_players
        self._done = False

    def setup(self):
        pass

    def play_sequence(self, stats_tracker=None):
        seq = Sequence(num_players=self.num_players)
        # 5 steps
        for i in range(6):
            seq.append(
                observation=np.ones(4) * (self.worker_id + i),
                terminated=(i == 5),
                truncated=False,
                action=0 if i < 5 else None,
                reward=1.0 if i < 5 else None,
                player_id=0,
            )
        seq.duration_seconds = 0.1
        return seq


def test_shared_memory_ipc():
    network = MockNetwork()
    config = MockConfig()
    executor = TorchMPExecutor()

    worker_args = (
        None,  # env_factory
        network,
        None,  # selector
        1,  # num_players
        config,
        torch.device("cpu"),
        "test_executor",
    )

    print("Launching workers...")
    executor.launch(MockActor, worker_args, num_workers=2)

    print("Collecting data...")
    # Collect 4 sequences
    data, stats = executor.collect_data(min_samples=4)

    assert len(data) >= 4
    for i, seq in enumerate(data):
        assert isinstance(seq, Sequence)
        assert len(seq.observation_history) == 6
        assert len(seq.action_history) == 5
        assert len(seq.rewards) == 5
        # Verify data integrity (obs should be ones * some value)
        obs = np.array(seq.observation_history)
        assert np.all(obs >= 0)
        print(f"Sequence {i} passed integrity check.")

    executor.stop()
    print("Test passed!")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    import platform

    if platform.system() == "Darwin":
        try:
            mp.set_sharing_strategy("file_system")
        except Exception:
            pass
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    try:
        test_shared_memory_ipc()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
