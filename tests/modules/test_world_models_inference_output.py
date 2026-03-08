import pytest
import torch
from modules.world_models.inference_output import MuZeroNetworkState

pytestmark = pytest.mark.unit


def test_muzero_network_state_batch_unbatch_none():
    """Verifies batching logic when there is no recurrent memory."""
    state1 = MuZeroNetworkState(dynamics=torch.randn(1, 4), wm_memory=None)
    state2 = MuZeroNetworkState(dynamics=torch.randn(1, 4), wm_memory=None)

    batched = MuZeroNetworkState.batch([state1, state2])
    assert batched.dynamics.shape == (2, 4)
    assert batched.wm_memory is None

    unbatched = batched.unbatch()
    assert len(unbatched) == 2
    assert unbatched[0].wm_memory is None


def test_muzero_network_state_batch_unbatch_tensor():
    """Verifies batching logic for standard tensor recurrent memory."""
    state1 = MuZeroNetworkState(dynamics=torch.randn(1, 4), wm_memory=torch.randn(1, 8))
    state2 = MuZeroNetworkState(dynamics=torch.randn(1, 4), wm_memory=torch.randn(1, 8))

    batched = MuZeroNetworkState.batch([state1, state2])
    assert batched.wm_memory.shape == (2, 8)

    unbatched = batched.unbatch()
    assert unbatched[0].wm_memory.shape == (1, 8)
    assert torch.allclose(unbatched[0].wm_memory, state1.wm_memory)


def test_muzero_network_state_batch_unbatch_tuple_lstm_style():
    """Verifies complex LSTM batching across dim=1."""
    # Standard PyTorch LSTM state shapes: (Num_Layers, Batch, Hidden)
    h1, c1 = torch.randn(1, 1, 8), torch.randn(1, 1, 8)
    h2, c2 = torch.randn(1, 1, 8), torch.randn(1, 1, 8)

    state1 = MuZeroNetworkState(dynamics=torch.randn(1, 4), wm_memory=(h1, c1))
    state2 = MuZeroNetworkState(dynamics=torch.randn(1, 4), wm_memory=(h2, c2))

    batched = MuZeroNetworkState.batch([state1, state2])
    bh, bc = batched.wm_memory

    # Should be stacked along the batch dimension (dim 1 for LSTM)
    assert bh.shape == (1, 2, 8)

    unbatched = batched.unbatch()
    uh, uc = unbatched[0].wm_memory
    assert uh.shape == (1, 1, 8)
    assert torch.allclose(uh, h1)


def test_muzero_network_state_batch_unbatch_dict():
    """Verifies multi-modal dictionary state batching."""
    d1 = {"feat": torch.randn(1, 4), "empty": None, "lstm": torch.randn(1, 1, 8)}
    d2 = {"feat": torch.randn(1, 4), "empty": None, "lstm": torch.randn(1, 1, 8)}

    state1 = MuZeroNetworkState(dynamics=torch.randn(1, 4), wm_memory=d1)
    state2 = MuZeroNetworkState(dynamics=torch.randn(1, 4), wm_memory=d2)

    batched = MuZeroNetworkState.batch([state1, state2])
    assert batched.wm_memory["feat"].shape == (2, 4)
    assert batched.wm_memory["empty"] is None
    assert batched.wm_memory["lstm"].shape == (1, 2, 8)

    unbatched = batched.unbatch()
    assert unbatched[1].wm_memory["feat"].shape == (1, 4)
    assert unbatched[1].wm_memory["empty"] is None
    assert unbatched[1].wm_memory["lstm"].shape == (1, 1, 8)
