import unittest
import torch

from stats.stats import StatTracker

class TestStatTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = StatTracker("test_model")

    def test_append_scalar(self):
        """Test appending simple scalar values (backward compatibility)."""
        self.tracker.append("scalar_metric", 1.0)
        self.tracker.append("scalar_metric", 2.0)
        
        expected_shape = (2,)
        self.assertEqual(self.tracker.stats["scalar_metric"].shape, expected_shape)
        self.assertTrue(torch.equal(self.tracker.stats["scalar_metric"], torch.tensor([1.0, 2.0])))

    def test_append_tensor_2d(self):
        """Test appending 2D tensors (e.g. chance probabilities)."""
        # Simulate appending (1, 5) tensors twice
        tensor1 = torch.randn(1, 5)
        tensor2 = torch.randn(1, 5)
        
        self.tracker.append("tensor_metric", tensor1)
        self.tracker.append("tensor_metric", tensor2)
        
        expected_shape = (2, 5)
        self.assertEqual(self.tracker.stats["tensor_metric"].shape, expected_shape)
        
        # Verify content matches
        expected_tensor = torch.cat((tensor1, tensor2), dim=0)
        self.assertTrue(torch.equal(self.tracker.stats["tensor_metric"], expected_tensor))

    def test_append_mismatched_shapes(self):
        """Test that appending tensors with different shapes raises an error."""
        self.tracker.append("mismatch_metric", torch.randn(1, 5))
        # PyTorch cat will raise an error if dimensions don't match, which is expected behavior
        with self.assertRaises(RuntimeError):
            self.tracker.append("mismatch_metric", torch.randn(1, 3))

if __name__ == "__main__":
    unittest.main()
