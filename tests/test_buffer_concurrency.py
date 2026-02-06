import torch
import torch.multiprocessing as mp
import numpy as np
import unittest
from replay_buffers.modular_buffer import ModularReplayBuffer, BufferConfig
from replay_buffers.concurrency import LocalBackend, TorchMPBackend


class TestBufferConcurrency(unittest.TestCase):
    def setUp(self):
        self.max_size = 100
        self.batch_size = 10
        self.configs = [
            BufferConfig("observations", shape=(4,), dtype=torch.float32),
            BufferConfig("actions", shape=(), dtype=torch.int64),
        ]

    def test_local_backend(self):
        backend = LocalBackend()
        buffer = ModularReplayBuffer(
            max_size=self.max_size,
            buffer_configs=self.configs,
            batch_size=self.batch_size,
            backend=backend,
        )

        # Test basic storage
        obs = np.random.rand(4).astype(np.float32)
        idx = buffer.store(observations=obs, actions=1)
        self.assertEqual(idx, 0)
        self.assertEqual(buffer.size, 1)

        # Test sampling
        # Fill buffer to batch size
        for _ in range(self.batch_size):
            buffer.store(observations=obs, actions=1)

        batch = buffer.sample()
        self.assertEqual(batch["observations"].shape, (self.batch_size, 4))
        self.assertFalse(batch["observations"].is_shared())

    def test_torch_mp_backend(self):
        backend = TorchMPBackend()
        buffer = ModularReplayBuffer(
            max_size=self.max_size,
            buffer_configs=self.configs,
            batch_size=self.batch_size,
            backend=backend,
        )

        self.assertTrue(buffer.is_shared)
        self.assertTrue(buffer.buffers["observations"].is_shared())

        # Test storage with lock
        obs = np.random.rand(4).astype(np.float32)
        idx = buffer.store(observations=obs, actions=1)
        self.assertEqual(idx, 0)

        # Test sampling
        for _ in range(self.batch_size):
            buffer.store(observations=obs, actions=1)

        batch = buffer.sample()
        self.assertEqual(batch["observations"].shape, (self.batch_size, 4))
        # Samples should still be shared if they are views, but here they are likely gathered
        # StandardOutputProcessor usually returns tensors on the same device as buffers

    def _mp_worker(self, buffer, n_stores):
        for i in range(n_stores):
            buffer.store(observations=np.zeros(4, dtype=np.float32), actions=i)

    def test_multiprocessing_integration(self):
        backend = TorchMPBackend()
        buffer = ModularReplayBuffer(
            max_size=1000,
            buffer_configs=self.configs,
            batch_size=self.batch_size,
            backend=backend,
        )

        num_workers = 4
        stores_per_worker = 100
        processes = []

        for _ in range(num_workers):
            p = mp.Process(target=self._mp_worker, args=(buffer, stores_per_worker))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        self.assertEqual(buffer.size, num_workers * stores_per_worker)


if __name__ == "__main__":
    unittest.main()
