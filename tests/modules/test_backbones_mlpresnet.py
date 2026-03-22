import pytest
import torch
from torch import nn
from modules.backbones.mlpresnet import MLPResNetBackbone
from configs.modules.backbones.mlpresnet import MLPResNetConfig

pytestmark = pytest.mark.unit


def test_mlpresnet_forward_4d_flattening():
    """Verifies that 4D inputs (e.g., images) are flattened automatically."""
    torch.manual_seed(42)
    # Pass configuration as a single dictionary
    config = MLPResNetConfig({"widths": [32, 32]})
    config.noisy_sigma = 0.0
    config.norm_type = "none"
    config.activation = nn.ReLU()

    backbone = MLPResNetBackbone(config, input_shape=(2, 3, 8, 8))
    x = torch.randn(2, 3, 8, 8)
    out = backbone(x)
    assert out.shape == (2, 32)


def test_mlpresnet_width_projection_and_noise_reset():
    """Tests projection layers when width changes and verifies reset_noise executes safely."""
    torch.manual_seed(42)
    config = MLPResNetConfig({"widths": [16, 32]})
    config.noisy_sigma = 0.5
    config.norm_type = "none"
    config.activation = nn.ReLU()

    backbone = MLPResNetBackbone(config, input_shape=(2, 16))
    x = torch.randn(2, 16)
    out1 = backbone(x)

    # NOTE: MLPResidualBlock now exposes a reset_noise() method.
    # We assert that the routing does not crash when called and noise is reset.
    backbone.reset_noise()
    out2 = backbone(x)

    assert out1.shape == (2, 32)
    # Since it's noisy, out1 and out2 should be different with the same seed if noise was reset between them
    # But wait, torch.manual_seed(42) was only called at the start.
    # MLPResidualBlock uses global torch.randn without its own seed.
    # If noise was reset, the weights changed.
    assert not torch.allclose(out1, out2)


def test_mlpresnet_sequential_structure():
    """Verifies that the backbone uses nn.Sequential for layer management."""
    torch.manual_seed(42)
    config = MLPResNetConfig({"widths": [16, 16]})
    config.noisy_sigma = 0.0
    config.norm_type = "none"
    config.activation = nn.ReLU()

    backbone = MLPResNetBackbone(config, input_shape=(2, 16))
    
    # Check that 'model' attribute exists and is nn.Sequential
    assert hasattr(backbone, "model")
    assert isinstance(backbone.model, nn.Sequential)
    
    # With widths [16, 16], we expect two MLPResidualBlocks (no projection needed as initial_width=16)
    from modules.backbones.mlpresnet import MLPResidualBlock
    assert len(backbone.model) == 2
    for layer in backbone.model:
        assert isinstance(layer, MLPResidualBlock)
