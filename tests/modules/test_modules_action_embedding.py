import pytest
import torch

from modules.embeddings.action_embedding import ActionEncoder

pytestmark = pytest.mark.unit


class TestActionEncoder:
    """Test ActionEncoder from modules/embeddings/action_embedding.py."""

    def test_action_encoder_discrete_image_path(self):
        torch.manual_seed(42)
        action_space_size = 5
        embedding_dim = 32
        encoder = ActionEncoder(
            action_space_size=action_space_size,
            embedding_dim=embedding_dim,
        )

        batch_size = 4
        # Expecting (B, A)
        action = torch.randn(batch_size, action_space_size)
        target_shape = (batch_size, 1, 8, 8) # Channel in target doesn't matter, we match H, W

        output = encoder(action, target_shape)
        assert output.shape == (batch_size, embedding_dim, 8, 8)

    def test_action_encoder_vector_path(self):
        torch.manual_seed(42)
        action_space_size = 5
        embedding_dim = 32
        encoder = ActionEncoder(
            action_space_size=action_space_size,
            embedding_dim=embedding_dim,
        )

        batch_size = 4
        action_indices = torch.randint(0, action_space_size, (batch_size,))
        action = torch.nn.functional.one_hot(action_indices, num_classes=action_space_size).float()
        target_shape = (batch_size, 100) # Only ndim matters

        output = encoder(action, target_shape)
        assert output.shape == (batch_size, embedding_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_action_encoder_invalid_target_shape(self):
        torch.manual_seed(42)
        action_space_size = 5
        embedding_dim = 32
        encoder = ActionEncoder(
            action_space_size=action_space_size,
            embedding_dim=embedding_dim,
        )

        batch_size = 4
        action = torch.randn(batch_size, action_space_size)
        invalid_target_shape = (batch_size, embedding_dim, 1) # ndim 3

        with pytest.raises(ValueError, match="must be length 2 or 4"):
            encoder(action, invalid_target_shape)

    def test_action_encoder_continuous_image_path(self):
        torch.manual_seed(42)
        action_space_size = 3
        embedding_dim = 32
        encoder = ActionEncoder(
            action_space_size=action_space_size,
            embedding_dim=embedding_dim,
        )

        batch_size = 4
        action = torch.randn(batch_size, action_space_size)
        target_shape = (batch_size, 64, 8, 8)

        output = encoder(action, target_shape)
        assert output.shape == (batch_size, embedding_dim, 8, 8)

    def test_action_encoder_output_no_nans(self):
        torch.manual_seed(42)
        action_space_size = 5
        embedding_dim = 32
        encoder = ActionEncoder(
            action_space_size=action_space_size,
            embedding_dim=embedding_dim,
        )

        batch_size = 4
        action = torch.randn(batch_size, action_space_size)
        target_shape = (batch_size, 1, 8, 8)

        output = encoder(action, target_shape)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
