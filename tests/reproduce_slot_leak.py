import torch
import torch.nn as nn
import numpy as np
import time
import os
import platform
import torch.multiprocessing as mp
from typing import Optional, Any, Tuple
from agents.executors.torch_mp_executor import TorchMPExecutor
from replay_buffers.sequence import Sequence
from queue import Empty

# Set sharing strategy for Mac
if platform.system() == "Darwin":
    try:
        mp.set_sharing_strategy("file_system")
    except Exception:
        pass


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

    def make_env(self):
        return None


class MockConfig:
    def __init__(self, num_workers=2, pool_size=None):
        self.max_episode_length = 10
        self.multi_process = True
        self.num_workers = num_workers
        self.game = MockGame()
        if pool_size:
            self.shared_memory_pool_size = pool_size


class SlowActor:
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

    def setup(self):
        pass

    def play_sequence(self, stats_tracker=None):
        # Generate some data
        seq = Sequence(num_players=self.num_players)
        for i in range(2):
            seq.append(
                observation=np.ones(4) * (self.worker_id + i),
                terminated=(i == 1),
                truncated=False,
                action=0 if i < 1 else None,
                reward=1.0 if i < 1 else None,
                player_id=0,
            )
        # Sleep to simulate some work so we don't flood the queue TOO instantly
        time.sleep(0.01)
        return seq


def reproduce_leak():
    print("Testing slot leak fix and increased capacity...")
    network = MockNetwork()
    num_workers = 4
    config = MockConfig(num_workers=num_workers)
    executor = TorchMPExecutor()

    worker_args = (None, network, None, 1, config, torch.device("cpu"), "test_leak")

    print(f"Launching {num_workers} workers...")
    executor.launch(SlowActor, worker_args, num_workers=num_workers)

    # Wait for some data to be produced and placed in slots
    time.sleep(1.0)

    # Check pool size
    num_slots = executor.shared_pool.num_slots
    expected_default_slots = max(num_workers * 4, 64)
    print(f"Default num_slots: {num_slots} (Expected: {expected_default_slots})")
    assert (
        num_slots == expected_default_slots
    ), f"Expected {expected_default_slots} slots, got {num_slots}"

    # Stop executor while things are likely in flight or in result_queue
    print("Stopping executor with data in flight...")
    executor.stop()

    # Wait for stop and draining
    time.sleep(2.0)

    # All slots should have been released
    free_slots_remaining = executor.shared_pool.free_slots.qsize()
    print(f"Free slots after stop: {free_slots_remaining} / {num_slots}")

    assert (
        free_slots_remaining == num_slots
    ), f"Slot leak detected! {num_slots - free_slots_remaining} slots not returned."
    print("Slot leak verification PASSED!")


def test_config_override():
    print("\nTesting configuration override for pool size...")
    network = MockNetwork()
    pool_size = 123
    config = MockConfig(num_workers=2, pool_size=pool_size)
    executor = TorchMPExecutor()
    worker_args = (None, network, None, 1, config, torch.device("cpu"), "test_override")

    executor.launch(SlowActor, worker_args, num_workers=2)
    num_slots = executor.shared_pool.num_slots
    print(f"Overridden num_slots: {num_slots} (Expected: {pool_size})")
    assert num_slots == pool_size, f"Expected {pool_size} slots, got {num_slots}"
    executor.stop()
    print("Config override verification PASSED!")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    try:
        reproduce_leak()
        test_config_override()
        print("\nAll verifications PASSED!")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
