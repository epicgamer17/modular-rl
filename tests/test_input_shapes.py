import unittest
import torch
import torch.nn as nn
from modules.backbones.conv import ConvBackbone
from modules.backbones.dense import DenseBackbone
from modules.backbones.resnet import ResNetBackbone
from configs.modules.backbones.conv import ConvConfig
from configs.modules.backbones.dense import DenseConfig
from configs.modules.backbones.resnet import ResNetConfig
from modules.heads.base import BaseHead
from configs.modules.architecture_config import ArchitectureConfig
from modules.heads.strategies import OutputStrategy


class MockStrategy(OutputStrategy):
    @property
    def num_bins(self):
        return 10

    def logits_to_vector(self, logits):
        return logits

    def logits_to_scalar(self, logits):
        return logits

    def logits_to_probs(self, logits):
        return logits

    def scalar_to_target(self, scalar):
        return scalar

    def compute_loss(self, prediction, target):
        return 0.0

    def get_distribution(self, prediction):
        return None

    def to_expected_value(self, prediction):
        return prediction


class MockHead(BaseHead):
    pass


class TestInputShapes(unittest.TestCase):
    def test_conv_backbone(self):
        # New standard: (C, H, W) input shape (no batch dim)
        input_shape = (3, 32, 32)
        config_dict = {
            "filters": [16, 32],
            "kernel_sizes": [3, 3],
            "strides": [1, 2],
            "activation": "relu",
            "norm": "none",
            "backbone_type": "conv",
        }
        config = ConvConfig(config_dict)
        backbone = ConvBackbone(config, input_shape)

        # Test input shape
        x = torch.randn(2, *input_shape)  # Batch of 2
        out = backbone(x)

        # Output shape check: (B, 32, 16, 16) - based on strides and filters
        # Layer 1: 3->16, stride 1. 32x32 -> 32x32
        # Layer 2: 16->32, stride 2. 32x32 -> 16x16
        self.assertEqual(out.shape, (2, 32, 16, 16))

        # output_shape property check: (32, 16, 16)
        self.assertEqual(backbone.output_shape, (32, 16, 16))

    def test_dense_backbone(self):
        # New standard: (D,) input shape
        input_shape = (64,)
        config_dict = {
            "widths": [128, 64],
            "activation": "relu",
            "norm": "none",
            "backbone_type": "dense",
        }
        config = DenseConfig(config_dict)
        backbone = DenseBackbone(config, input_shape)

        # Test input
        x = torch.randn(2, 64)  # Batch of 2
        out = backbone(x)

        self.assertEqual(out.shape, (2, 64))
        self.assertEqual(backbone.output_shape, (64,))

    def test_resnet_backbone(self):
        # New standard: (C, H, W) -> (3, 64, 64)
        input_shape = (3, 64, 64)
        config_dict = {
            "blocks_per_group": [1, 1],
            "filters": [16, 32],
            "kernel_sizes": [3, 3],
            "strides": [1, 2],
            "activation": "relu",
            "norm": "none",
            "backbone_type": "resnet",
        }
        config = ResNetConfig(config_dict)
        backbone = ResNetBackbone(config, input_shape)

        x = torch.randn(2, *input_shape)
        out = backbone(x)

        # Layer 1: stride 1 -> 64x64
        # Layer 2: stride 2 -> 32x32
        self.assertEqual(out.shape, (2, 32, 32, 32))
        self.assertEqual(backbone.output_shape, (32, 32, 32))

    def test_head_flat_dim(self):
        input_shape = (64,)  # Feature vector from backbone
        arch_config = ArchitectureConfig({"noisy_sigma": 0})
        strategy = MockStrategy()

        head = MockHead(arch_config, input_shape, strategy)

        # Should be 64 (product of dimensions)
        self.assertEqual(head.flat_dim, 64)

        x = torch.randn(5, 64)  # Batch 5
        out, _ = head(x)
        self.assertEqual(out.shape, (5, 10))


if __name__ == "__main__":
    unittest.main()
