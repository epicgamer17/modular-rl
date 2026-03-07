import pytest
import torch
from torch import nn

from modules.utils import (
    _normalize_hidden_state,
    build_normalization_layer,
    calculate_same_padding,
    get_clean_state_dict,
    get_uncompiled_model,
    prepare_activations,
    scalar_to_support,
    scale_gradient,
    support_to_scalar,
)

pytestmark = pytest.mark.unit


def test_modules_utils_support_round_trip_batch_values():
    x = torch.tensor([-2.0, 0.0, 3.5, 10.0], dtype=torch.float32)
    support = scalar_to_support(x, support_size=10)
    recovered = support_to_scalar(support, support_size=10)

    assert support.shape == (4, 21)
    assert recovered.shape == (4,)
    assert torch.allclose(recovered, x, atol=1e-4, rtol=1e-4)


def test_modules_utils_support_to_scalar_accepts_1d_distribution():
    distribution = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    scalar = support_to_scalar(distribution, support_size=1)

    assert scalar.shape == ()
    assert scalar.item() == pytest.approx(0.0)


def test_modules_utils_calculate_same_padding_branches():
    manual_padding, torch_padding = calculate_same_padding((6, 6), (3, 3), (2, 2))
    assert manual_padding == (0, 1, 0, 1)
    assert torch_padding is None

    manual_padding, torch_padding = calculate_same_padding((9, 9), (3, 3), (2, 2))
    assert manual_padding is None
    assert torch_padding == (1, 1)

    manual_padding, torch_padding = calculate_same_padding((5, 5), (3, 3), 1)
    assert manual_padding is None
    assert torch_padding == "same"


def test_modules_utils_prepare_activations_and_error_path():
    relu = prepare_activations("relu")
    identity = prepare_activations("linear")
    passthrough = prepare_activations(nn.Tanh())

    assert isinstance(relu, nn.ReLU)
    assert isinstance(identity, nn.Identity)
    assert isinstance(passthrough, nn.Tanh)

    with pytest.raises(ValueError, match="not recognized"):
        prepare_activations("not-an-activation")


def test_modules_utils_build_normalization_layer_variants():
    assert isinstance(build_normalization_layer("batch", 8, dim=2), nn.BatchNorm2d)
    assert isinstance(build_normalization_layer("batch", 8, dim=1), nn.BatchNorm1d)
    assert isinstance(build_normalization_layer("layer", 8, dim=1), nn.LayerNorm)
    assert isinstance(build_normalization_layer("none", 8, dim=1), nn.Identity)

    with pytest.raises(ValueError, match="Unknown normalization type"):
        build_normalization_layer("invalid", 8, dim=1)


def test_modules_utils_scale_gradient_applies_backward_scaling():
    x = torch.tensor([2.0], requires_grad=True)
    y = scale_gradient(x, scale=0.25)
    y.backward()

    assert y.item() == pytest.approx(2.0)
    assert x.grad is not None
    assert x.grad.item() == pytest.approx(0.25)


def test_modules_utils_normalize_hidden_state_for_dense_and_spatial():
    dense = torch.tensor([[1.0, 3.0], [2.0, 2.0]], dtype=torch.float32)
    dense_norm = _normalize_hidden_state(dense)
    assert torch.allclose(dense_norm[0], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(dense_norm[1], torch.tensor([0.0, 0.0]), atol=1e-6)

    spatial = torch.tensor([[[[1.0, 3.0], [2.0, 4.0]]]], dtype=torch.float32)
    spatial_norm = _normalize_hidden_state(spatial)
    assert spatial_norm.shape == spatial.shape
    assert torch.isclose(spatial_norm.min(), torch.tensor(0.0))
    assert torch.isclose(spatial_norm.max(), torch.tensor(1.0))


def test_modules_utils_state_dict_helpers_strip_prefix_and_copy_model():
    class PrefixModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(2, 2)

        def state_dict(self, *args, **kwargs):
            return {
                "_orig_mod.layer.weight": torch.tensor([1.0]),
                "layer.bias": torch.tensor([0.0]),
            }

    model = PrefixModel()
    cleaned = get_clean_state_dict(model)
    assert "_orig_mod.layer.weight" not in cleaned
    assert "layer.weight" in cleaned
    assert "layer.bias" in cleaned

    class CompiledLikeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def obs_inference(self, x):
            return x

    compiled_like = CompiledLikeModel()
    compiled_like.obs_inference = lambda x: x + 1
    uncompiled = get_uncompiled_model(compiled_like)

    assert uncompiled is not compiled_like
    assert "obs_inference" in compiled_like.__dict__
    assert "obs_inference" not in uncompiled.__dict__
