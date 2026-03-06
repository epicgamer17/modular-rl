import pytest
pytestmark = pytest.mark.unit

import torch
from modules.blocks.conv import Conv2dStack, ConvTranspose2dStack
from modules.blocks.dense import DenseStack


def test_bias_rule():
    # Test Conv2dStack
    stack = Conv2dStack(
        input_shape=(3, 32, 32),
        filters=[16],
        kernel_sizes=[3],
        strides=[1],
        norm_type="batch",
    )
    conv = stack._layers[0][0]
    assert (
        conv.bias is None
    ), "Conv2dStack should have bias=False when norm_type='batch'"

    stack_no_norm = Conv2dStack(
        input_shape=(3, 32, 32),
        filters=[16],
        kernel_sizes=[3],
        strides=[1],
        norm_type="none",
    )
    # If norm is none, is it still in a Sequential?
    # Yes, Conv2dStack always puts layers in self._layers as Sequential or individual.
    layer = stack_no_norm._layers[0]
    if isinstance(layer, torch.nn.Sequential):
        # Find the conv2d in there
        conv_no_norm = next(m for m in layer if isinstance(m, torch.nn.Conv2d))
    else:
        conv_no_norm = layer
    assert (
        conv_no_norm.bias is not None
    ), "Conv2dStack should have bias=True when norm_type='none'"

    # Test ConvTranspose2dStack
    tstack = ConvTranspose2dStack(
        input_shape=(16, 8, 8),
        filters=[8],
        kernel_sizes=[3],
        strides=[2],
        norm_type="layer",
    )
    tconv = tstack._layers[0][0]
    assert (
        tconv.bias is None
    ), "ConvTranspose2dStack should have bias=False when norm_type='layer'"

    # Test DenseStack
    dstack = DenseStack(initial_width=10, widths=[20], norm_type="batch")
    # DenseStack structure: Sequential(Dense/NoisyDense, Norm)
    dense = dstack._layers[0][0].layer
    assert (
        dense.bias is None
    ), "DenseStack should have bias=False when norm_type='batch'"

    # Verify norm layer exists
    norm = dstack._layers[0][1]
    assert isinstance(
        norm, torch.nn.BatchNorm1d
    ), f"Expected BatchNorm1d, got {type(norm)}"

    print("✅ All block compliance tests passed!")


if __name__ == "__main__":
    test_bias_rule()
