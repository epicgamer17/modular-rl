import torch
import unittest
from search.search_py.min_max_stats import MinMaxStats as PyMinMaxStats
from search.aos_search.min_max_stats import VectorizedMinMaxStats as AOSMinMaxStats


class TestSoftMinMax(unittest.TestCase):
    def test_py_search_soft_min_max(self):
        # Hard min-max (default epsilon)
        stats_hard = PyMinMaxStats(known_bounds=[0.0, 0.1], epsilon=1e-8)
        # Soft min-max (EfficientZero epsilon)
        stats_soft = PyMinMaxStats(known_bounds=[0.0, 0.1], epsilon=0.01)

        val = 0.05
        norm_hard = stats_hard.normalize(val)
        norm_soft = stats_soft.normalize(val)

        print(f"PySearch Hard: {norm_hard}, Soft: {norm_soft}")
        # (0.05 - 0.0) / 0.1 = 0.5
        self.assertAlmostEqual(norm_hard, 0.5)
        self.assertAlmostEqual(norm_soft, 0.5)

        # Now test when range is smaller than epsilon
        stats_hard_small = PyMinMaxStats(known_bounds=[0.0, 0.001], epsilon=1e-8)
        stats_soft_small = PyMinMaxStats(known_bounds=[0.0, 0.001], epsilon=0.01)

        val_small = 0.0005
        norm_hard_small = stats_hard_small.normalize(val_small)
        norm_soft_small = stats_soft_small.normalize(val_small)

        print(f"PySearch Small Range Hard: {norm_hard_small}, Soft: {norm_soft_small}")
        # Hard: (0.0005 - 0.0) / 0.001 = 0.5
        # Soft: (0.0005 - 0.0) / 0.01 = 0.05
        self.assertAlmostEqual(norm_hard_small, 0.5)
        self.assertAlmostEqual(norm_soft_small, 0.05)

    def test_aos_search_soft_min_max(self):
        device = torch.device("cpu")
        # Hard min-max
        stats_hard = AOSMinMaxStats.allocate(
            batch_size=1, device=device, known_bounds=[0.0, 0.1], epsilon=1e-8
        )
        # Soft min-max
        stats_soft = AOSMinMaxStats.allocate(
            batch_size=1, device=device, known_bounds=[0.0, 0.1], epsilon=0.01
        )

        val = torch.tensor([0.05])
        norm_hard = stats_hard.normalize(val)
        norm_soft = stats_soft.normalize(val)

        print(f"AOS Hard: {norm_hard.item()}, Soft: {norm_soft.item()}")
        self.assertAlmostEqual(norm_hard.item(), 0.5)
        self.assertAlmostEqual(norm_soft.item(), 0.5)

        # Test small range
        stats_hard_small = AOSMinMaxStats.allocate(
            batch_size=1, device=device, known_bounds=[0.0, 0.001], epsilon=1e-8
        )
        stats_soft_small = AOSMinMaxStats.allocate(
            batch_size=1, device=device, known_bounds=[0.0, 0.001], epsilon=0.01
        )

        val_small = torch.tensor([0.0005])
        norm_hard_small = stats_hard_small.normalize(val_small)
        norm_soft_small = stats_soft_small.normalize(val_small)

        print(
            f"AOS Small Range Hard: {norm_hard_small.item()}, Soft: {norm_soft_small.item()}"
        )
        self.assertAlmostEqual(norm_hard_small.item(), 0.5)
        self.assertAlmostEqual(norm_soft_small.item(), 0.05)


if __name__ == "__main__":
    unittest.main()
