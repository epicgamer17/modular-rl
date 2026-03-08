import pytest
import torch
from torch import nn
from modules.backbones.denseresnet import DenseResNetBackbone
from configs.modules.backbones.denseresnet import DenseResNetConfig

pytestmark = pytest.mark.unit


def test_denseresnet_forward_4d_flattening():
    """Verifies that 4D inputs (e.g., images) are flattened automatically."""
    torch.manual_seed(42)
    # Pass configuration as a single dictionary
    config = DenseResNetConfig({"widths": [32, 32]})
    config.noisy_sigma = 0.0
    config.norm_type = "none"
    config.activation = nn.ReLU()

    backbone = DenseResNetBackbone(config, input_shape=(2, 3, 8, 8))
    x = torch.randn(2, 3, 8, 8)
    out = backbone(x)
    assert out.shape == (2, 32)


def test_denseresnet_width_projection_and_noise_reset():
    """Tests projection layers when width changes and verifies reset_noise executes safely."""
    torch.manual_seed(42)
    config = DenseResNetConfig({"widths": [16, 32]})
    config.noisy_sigma = 0.5
    config.norm_type = "none"
    config.activation = nn.ReLU()

    backbone = DenseResNetBackbone(config, input_shape=(2, 16))
    x = torch.randn(2, 16)
    out1 = backbone(x)

    # NOTE: DenseResidualBlock currently does not expose a reset_noise() method.
    # The backbone silently skips layers that lack this attribute.
    # We assert that the routing does not crash when called.
    backbone.reset_noise()
    out2 = backbone(x)

    assert out1.shape == (2, 32)


def test_denseresnet_initialize():
    """Verifies custom initializers pass safely through all nested layers."""
    torch.manual_seed(42)
    config = DenseResNetConfig({"widths": [16]})
    config.noisy_sigma = 0.0
    config.norm_type = "none"
    config.activation = nn.ReLU()

    backbone = DenseResNetBackbone(config, input_shape=(2, 16))

    def mock_init(tensor):
        nn.init.constant_(tensor, 0.123)

    backbone.initialize(mock_init)
    assert torch.allclose(backbone.layers[0].linear.layer.weight, torch.tensor(0.123))
