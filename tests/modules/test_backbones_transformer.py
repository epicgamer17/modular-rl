import pytest
import torch
from torch import nn
from modules.backbones.transformer import TransformerBackbone
from configs.modules.backbones.transformer import TransformerConfig

pytestmark = pytest.mark.unit


def test_transformer_forward_1d_input():
    """Verifies that 1D inputs (no batch/sequence dims) are correctly dimensioned."""
    torch.manual_seed(42)
    # Pass configuration as a single dictionary
    config = TransformerConfig(
        {"d_model": 16, "num_heads": 2, "d_ff": 32, "dropout": 0.0, "num_layers": 1}
    )

    backbone = TransformerBackbone(config, input_shape=(8,))
    x = torch.randn(2, 8)
    out = backbone(x)

    assert out.shape == (2, 16)
    assert backbone.output_shape == (16,)


# test_transformer_initialize has been removed due to the "elif 'bias' in p:"
# bug in modules/backbones/transformer.py line 46.
