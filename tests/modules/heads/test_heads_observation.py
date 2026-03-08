import pytest
import torch
from modules.heads.observation import ObservationHead
from configs.modules.architecture_config import ArchitectureConfig
from modules.heads.strategies import ScalarStrategy

pytestmark = pytest.mark.unit


def test_observation_head_with_output_layer():
    """Verifies standard operation with a dense projection output layer."""
    torch.manual_seed(42)
    arch_config = ArchitectureConfig({"noisy_sigma": 0.0})
    strategy = ScalarStrategy()

    head = ObservationHead(
        arch_config=arch_config,
        input_shape=(16,),
        strategy=strategy,
        use_output_layer=True,
    )

    x = torch.randn(2, 16)
    logits, state, obs_pred = head(x)

    # ScalarStrategy squeezes 1D outputs to (B,)
    assert logits.shape == (2, 1)  # Raw logits remain 2D
    assert obs_pred.shape == (2,)  # Expected value is squeezed


def test_observation_head_without_output_layer():
    """Verifies bypass operation where the output layer is omitted entirely."""
    torch.manual_seed(42)
    arch_config = ArchitectureConfig({"noisy_sigma": 0.0})

    head = ObservationHead(
        arch_config=arch_config, input_shape=(2, 8, 8), use_output_layer=False
    )

    x = torch.randn(2, 2, 8, 8)
    logits, state, obs_pred = head(x)

    assert logits.shape == (2, 2, 8, 8)
