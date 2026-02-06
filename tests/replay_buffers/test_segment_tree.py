import torch
import unittest
from replay_buffers.segment_tree import SumSegmentTree, MinSegmentTree
from replay_buffers.concurrency import LocalBackend, TorchMPBackend


class TestSegmentTree(unittest.TestCase):
    def test_sum_segment_tree(self):
        capacity = 8
        backend = LocalBackend()
        tree = SumSegmentTree(capacity, backend=backend)

        # Test basic setting and getting
        tree[0] = 1.0
        tree[1] = 2.0
        self.assertEqual(tree[0], 1.0)
        self.assertEqual(tree[1], 2.0)

        # Test sum operation
        self.assertEqual(tree.sum(0, 1), 3.0)
        self.assertEqual(tree.sum(0, 7), 3.0)

        tree[2] = 3.0
        self.assertEqual(tree.sum(0, 2), 6.0)
        self.assertEqual(tree.sum(1, 2), 5.0)

        # Test retrieve
        # tree values: [1, 2, 3, 0, 0, 0, 0, 0]
        # cumulative: [1, 3, 6, 6, 6, 6, 6, 6]
        self.assertEqual(tree.retrieve(0.5), 0)
        self.assertEqual(tree.retrieve(1.5), 1)
        self.assertEqual(tree.retrieve(3.5), 2)
        self.assertEqual(tree.retrieve(5.9), 2)

    def test_min_segment_tree(self):
        capacity = 8
        backend = LocalBackend()
        tree = MinSegmentTree(capacity, backend=backend)

        tree[0] = 10.0
        tree[1] = 5.0
        tree[2] = 20.0

        self.assertEqual(tree.min(0, 2), 5.0)
        self.assertEqual(tree.min(0, 0), 10.0)
        self.assertEqual(tree.min(2, 2), 20.0)

        self.assertEqual(tree.min_index(0, 2), 1 + capacity)  # Index in tree storage

    def test_torch_mp_backend(self):
        capacity = 4
        backend = TorchMPBackend()
        tree = SumSegmentTree(capacity, backend=backend)

        self.assertTrue(tree.tree.is_shared())
        tree[0] = 1.0
        self.assertEqual(tree.sum(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
