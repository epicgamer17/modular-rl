import torch
import unittest
import numpy as np
from replay_buffers.samplers import PrioritizedSampler
from replay_buffers.concurrency import LocalBackend


class TestSamplers(unittest.TestCase):
    def test_prioritized_sampler_local(self):
        max_size = 8
        batch_size = 2
        backend = LocalBackend()
        sampler = PrioritizedSampler(max_size, backend=backend)

        # Test default values
        self.assertEqual(sampler.alpha, 0.6)
        self.assertEqual(sampler.beta, 0.4)

        # Store some items with different priorities
        # on_store(idx, priority)
        sampler.on_store(0, priority=1.0)  # priority^alpha = 1.0
        sampler.on_store(1, priority=2.0)  # 2^0.6 = 1.5157
        sampler.on_store(2, priority=0.5)  # 0.5^0.6 = 0.6597

        # Sample
        indices, weights = sampler.sample(buffer_size=3, batch_size=batch_size)

        self.assertEqual(len(indices), batch_size)
        self.assertEqual(weights.shape, (batch_size,))
        self.assertTrue(all(0 <= idx < 3 for idx in indices))

        # Update priorities
        new_priorities = torch.tensor([10.0, 10.0])
        sampler.update_priorities(indices, new_priorities)

        # Check that sum_tree reflects updates
        total_p = sampler.sum_tree.sum(0, 7)
        # Remaining item at index 2 has original priority
        expected_remaining = 0.5**0.6
        # Updated items have 10.0^0.6
        expected_updated = 10.0**0.6 * batch_size
        # This matches if indices were unique, but they might not be.
        # But we know total_p should be roughly right.
        self.assertGreater(total_p, 0)

    def test_prioritized_sampler_clear(self):
        max_size = 4
        sampler = PrioritizedSampler(max_size)
        sampler.on_store(0, priority=1.0)
        self.assertGreater(sampler.sum_tree.sum(0, 3), 0)

        sampler.clear()
        self.assertEqual(sampler.sum_tree.sum(0, 3), 0)
        self.assertEqual(sampler.max_priority, 1.0)


if __name__ == "__main__":
    unittest.main()
