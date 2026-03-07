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
            is_continuous=False,
            single_action_plane=True,
        )

        batch_size = 4
        action = torch.randint(0, action_space_size, (batch_size, action_space_size))
        target_shape = (batch_size, embedding_dim, 8, 8)

        output = encoder(action, target_shape)
        assert output.shape == target_shape

    def test_action_encoder_discrete_vector_path(self):
        torch.manual_seed(42)
        action_space_size = 5
        embedding_dim = 32
        encoder = ActionEncoder(
            action_space_size=action_space_size,
            embedding_dim=embedding_dim,
            is_continuous=False,
            single_action_plane=True,
        )

        batch_size = 4
        action_indices = torch.randint(0, action_space_size, (batch_size,))
        action = torch.nn.functional.one_hot(action_indices, num_classes=action_space_size).float()
        target_shape = (batch_size, embedding_dim)

        output = encoder(action, target_shape)
        assert output.shape == target_shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_action_encoder_spatial_discrete(self):
        torch.manual_seed(42)
        action_space_size = 9
        embedding_dim = 16
        encoder = ActionEncoder(
            action_space_size=action_space_size,
            embedding_dim=embedding_dim,
            is_continuous=False,
            single_action_plane=False,
        )

        batch_size = 2
        action_indices = torch.randint(0, action_space_size, (batch_size,))
        action = torch.nn.functional.one_hot(action_indices, num_classes=action_space_size).float()
        target_shape = (batch_size, embedding_dim, 3, 3)

        output = encoder(action, target_shape)
        assert output.shape == target_shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_action_encoder_invalid_target_shape(self):
        torch.manual_seed(42)
        action_space_size = 5
        embedding_dim = 32
        encoder = ActionEncoder(
            action_space_size=action_space_size,
            embedding_dim=embedding_dim,
            is_continuous=False,
            single_action_plane=True,
        )

        batch_size = 4
        action_indices = torch.randint(0, action_space_size, (batch_size,))
        action = torch.nn.functional.one_hot(action_indices, num_classes=action_space_size).float()
        invalid_target_shape = (batch_size, embedding_dim, 1)

        with pytest.raises(ValueError, match="must be len 2"):
            encoder(action, invalid_target_shape)

    def test_action_encoder_continuous_image_path(self):
        torch.manual_seed(42)
        action_space_size = 3
        embedding_dim = 32
        encoder = ActionEncoder(
            action_space_size=action_space_size,
            embedding_dim=embedding_dim,
            is_continuous=True,
            single_action_plane=False,
        )

        batch_size = 4
        action = torch.randn(batch_size, action_space_size)
        target_shape = (batch_size, embedding_dim, 8, 8)

        output = encoder(action, target_shape)
        assert output.shape == target_shape

    def test_action_encoder_continuous_vector_path(self):
        torch.manual_seed(42)
        action_space_size = 3
        embedding_dim = 32
        encoder = ActionEncoder(
            action_space_size=action_space_size,
            embedding_dim=embedding_dim,
            is_continuous=True,
            single_action_plane=False,
        )

        batch_size = 4
        action = torch.randn(batch_size, action_space_size)
        target_shape = (batch_size, embedding_dim)

        output = encoder(action, target_shape)
        assert output.shape == target_shape

    def test_action_encoder_output_no_nans(self):
        torch.manual_seed(42)
        action_space_size = 5
        embedding_dim = 32
        encoder = ActionEncoder(
            action_space_size=action_space_size,
            embedding_dim=embedding_dim,
            is_continuous=False,
            single_action_plane=True,
        )

        batch_size = 4
        action = torch.randint(0, action_space_size, (batch_size, action_space_size))
        target_shape = (batch_size, embedding_dim, 8, 8)

        output = encoder(action, target_shape)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
