import torch
from torch import nn
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np

from modules.backbones.mlp import MLPBackbone

# Tier 1 Unit Test Marker
pytestmark = pytest.mark.unit

@st.composite
def batch_tensor_strategy(draw):
    """Generates a batch of tensors with random shapes and values."""
    batch_size = draw(st.integers(min_value=1, max_value=8))
    feature_dim = draw(st.integers(min_value=1, max_value=32))
    
    # Generate as numpy then convert to torch
    shape = (batch_size, feature_dim)
    data = draw(arrays(np.float32, shape, elements=st.floats(min_value=-10, max_value=10)))
    return torch.from_numpy(data)

@given(batch=batch_tensor_strategy())
def test_mlp_batch_consistency(batch):
    """
    Property: Batch Consistency.
    Output of a batch must be identical to the concatenation of outputs 
    for individual elements (assuming no BatchNorm).
    """
    input_dim = batch.shape[1]
    # Use LayerNorm or None to ensure batch independence
    model = MLPBackbone(input_shape=(input_dim,), widths=[16, 8], norm_type="layer")
    model.eval() # Important for deterministic behavior if there were dropout/batchnorm
    
    with torch.no_grad():
        # Full batch forward
        batch_out = model(batch)
        
        # Individual forwards
        individual_outs = []
        for i in range(batch.shape[0]):
            out = model(batch[i:i+1])
            individual_outs.append(out)
        cat_out = torch.cat(individual_outs, dim=0)
        
    torch.testing.assert_close(batch_out, cat_out, rtol=1e-5, atol=1e-5)

@given(batch=batch_tensor_strategy())
def test_mlp_permutation_invariance(batch):
    """
    Property: Batch Permutation Invariance.
    Shuffling the batch dimension must result in an identically shuffled output.
    """
    if batch.shape[0] < 2:
        return # Need at least 2 elements to permute
        
    input_dim = batch.shape[1]
    model = MLPBackbone(input_shape=(input_dim,), widths=[16, 8], norm_type="none")
    model.eval()
    
    # Create permutation
    perm = torch.randperm(batch.shape[0])
    permuted_batch = batch[perm]
    
    with torch.no_grad():
        orig_out = model(batch)
        perm_out = model(permuted_batch)
        
    torch.testing.assert_close(orig_out[perm], perm_out, rtol=1e-5, atol=1e-5)

@st.composite
def variable_shape_strategy(draw):
    """Generates valid MLP input shapes and corresponding tensors."""
    # MLPBackbone handles either (C,) or (C, H, W) via flattening
    is_3d = draw(st.booleans())
    batch_size = draw(st.integers(min_value=1, max_value=4))
    
    if is_3d:
        c = draw(st.integers(min_value=1, max_value=4))
        h = draw(st.integers(min_value=1, max_value=4))
        w = draw(st.integers(min_value=1, max_value=4))
        input_shape = (c, h, w)
        tensor_shape = (batch_size, c, h, w)
    else:
        c = draw(st.integers(min_value=1, max_value=16))
        input_shape = (c,)
        tensor_shape = (batch_size, c)
        
    data = draw(arrays(np.float32, tensor_shape, elements=st.floats(-1, 1)))
    return input_shape, torch.from_numpy(data)

@given(shape_data=variable_shape_strategy())
def test_mlp_shape_invariance(shape_data):
    """
    Property: Shape Invariance.
    The module must produce outputs with consistent rank and feature dimension
    regardless of whether the input is flat or spatial (if it supports flattening).
    """
    input_shape, batch = shape_data
    widths = [32, 16]
    model = MLPBackbone(input_shape=input_shape, widths=widths)
    
    out = model(batch)
    
    # Assert rank 2: [Batch, Output_Width]
    assert out.ndim == 2
    assert out.shape[0] == batch.shape[0]
    assert out.shape[1] == widths[-1]
