import pytest
pytestmark = pytest.mark.unit

def test_trainer_mp_import():
    import torch
    import torch.multiprocessing as mp

    assert torch is not None
    assert mp is not None
