import pytest
pytestmark = pytest.mark.integration

import torch
import torch.multiprocessing as mp
import numpy as np
from replay_buffers.modular_buffer import ModularReplayBuffer, BufferConfig
from replay_buffers.concurrency import LocalBackend, TorchMPBackend


MAX_SIZE = 100
BATCH_SIZE = 10
CONFIGS = [
    BufferConfig("observations", shape=(4,), dtype=torch.float32),
    BufferConfig("actions", shape=(), dtype=torch.int64),
]


def test_local_backend():
    backend = LocalBackend()
    buffer = ModularReplayBuffer(
        max_size=MAX_SIZE,
        buffer_configs=CONFIGS,
        batch_size=BATCH_SIZE,
        backend=backend,
    )

    # Test basic storage
    obs = np.random.rand(4).astype(np.float32)
    idx = buffer.store(observations=obs, actions=1)
    assert idx == 0
    assert buffer.size == 1

    # Test sampling
    # Fill buffer to batch size
    for _ in range(BATCH_SIZE):
        buffer.store(observations=obs, actions=1)

    batch = buffer.sample()
    assert batch["observations"].shape == (BATCH_SIZE, 4)
    assert not batch["observations"].is_shared()


def test_torch_mp_backend():
    backend = TorchMPBackend()
    try:
        buffer = ModularReplayBuffer(
            max_size=MAX_SIZE,
            buffer_configs=CONFIGS,
            batch_size=BATCH_SIZE,
            backend=backend,
        )
    except RuntimeError as err:
        if "Operation not permitted" in str(err) or "torch_shm_manager" in str(err):
            pytest.skip("Shared-memory backend is unavailable in this test environment")
        raise

    assert buffer.is_shared
    assert buffer.buffers["observations"].is_shared()

    # Test storage with lock
    obs = np.random.rand(4).astype(np.float32)
    idx = buffer.store(observations=obs, actions=1)
    assert idx == 0

    # Test sampling
    for _ in range(BATCH_SIZE):
        buffer.store(observations=obs, actions=1)

    batch = buffer.sample()
    assert batch["observations"].shape == (BATCH_SIZE, 4)
    # Samples should still be shared if they are views, but here they are likely gathered
    # StandardOutputProcessor usually returns tensors on the same device as buffers


def _mp_worker(buffer, n_stores):
    for i in range(n_stores):
        buffer.store(observations=np.zeros(4, dtype=np.float32), actions=i)


def test_multiprocessing_integration():
    backend = TorchMPBackend()
    try:
        buffer = ModularReplayBuffer(
            max_size=1000,
            buffer_configs=CONFIGS,
            batch_size=BATCH_SIZE,
            backend=backend,
        )
    except RuntimeError as err:
        if "Operation not permitted" in str(err) or "torch_shm_manager" in str(err):
            pytest.skip("Shared-memory backend is unavailable in this test environment")
        raise

    num_workers = 4
    stores_per_worker = 100
    processes = []

    for _ in range(num_workers):
        p = mp.Process(target=_mp_worker, args=(buffer, stores_per_worker))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    assert buffer.size == num_workers * stores_per_worker
