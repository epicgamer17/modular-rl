import pytest
import torch
import numpy as np
from replay_buffers.processors import GAEProcessor

pytestmark = pytest.mark.unit


def test_gae_processor_shape_mismatch():
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize directly with the required explicit parameters
    processor = GAEProcessor(gamma=0.99, gae_lambda=0.95)

    # Create dummy buffers with fundamentally incompatible shapes
    # e.g., rewards is 1D, but values is 2D, causing a broadcasting/cat crash
    invalid_buffers = {"rewards": torch.randn(10), "values": torch.randn(10, 5)}

    # We will try to process the first 5 elements
    trajectory_slice = slice(0, 5)

    # Verify it crashes safely during the tensor math inside finish_trajectory
    with pytest.raises(RuntimeError):
        processor.finish_trajectory(invalid_buffers, trajectory_slice)
