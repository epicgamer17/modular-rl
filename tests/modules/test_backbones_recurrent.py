import pytest
import torch
from torch import nn
from modules.backbones.recurrent import RecurrentBackbone
from configs.modules.backbones.recurrent import RecurrentConfig

pytestmark = pytest.mark.unit


def test_recurrent_backbone_gru_and_1d_input():
    torch.manual_seed(42)
    config = RecurrentConfig({"rnn_type": "gru", "hidden_size": 32, "num_layers": 1})
    net = RecurrentBackbone(config, input_shape=(16,))
    x_1d = torch.randn(2, 16)
    out, h_n = net(x_1d)
    assert out.shape == (2, 32)


def test_recurrent_backbone_lstm_and_3d_input():
    torch.manual_seed(42)
    config = RecurrentConfig({"rnn_type": "lstm", "hidden_size": 32, "num_layers": 1})
    net = RecurrentBackbone(config, input_shape=(10, 16))
    x_3d = torch.randn(2, 10, 16)
    out, h_n = net(x_3d)
    assert out.shape == (2, 32)


def test_recurrent_backbone_initialize():
    torch.manual_seed(42)
    config = RecurrentConfig({"rnn_type": "gru", "hidden_size": 16, "num_layers": 1})
    net = RecurrentBackbone(config, input_shape=(8,))

    def mock_init(tensor):
        nn.init.constant_(tensor, 0.5)

    net.initialize(mock_init)
    for name, param in net.rnn.named_parameters():
        if "weight" in name:
            assert torch.allclose(param, torch.full_like(param, 0.5))
        elif "bias" in name:
            assert torch.allclose(param, torch.zeros_like(param))
