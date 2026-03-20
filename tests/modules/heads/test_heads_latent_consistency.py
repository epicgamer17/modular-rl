import pytest
import torch
from configs.modules.architecture_config import ArchitectureConfig
from agents.learner.losses.representations import IdentityRepresentation
from modules.heads.latent_consistency import LatentConsistencyHead

pytestmark = pytest.mark.unit


def test_latent_consistency_head_custom_representation():
    """Verifies that the head projects to the expected dimension."""
    torch.manual_seed(42)
    arch_config = ArchitectureConfig({"noisy_sigma": 0.0})

    representation = IdentityRepresentation(num_features=256)

    head = LatentConsistencyHead(
        arch_config=arch_config,
        input_shape=(16,),
        representation=representation,
        projection_dim=256,
    )

    x = torch.randn(2, 16)
    logits, state, projected = head(x)
    assert projected.shape == (2, 256)
