import numpy as np
import pytest
import torch

from replay_buffers.modular_buffer import ModularReplayBuffer, BufferConfig

pytestmark = pytest.mark.unit


@pytest.fixture
def tiny_buffer_config():
    """Real configuration fixture for a tiny replay buffer."""
    return [
        BufferConfig(name="observations", shape=(4,), dtype=torch.float32, fill_value=0.0),
        BufferConfig(name="actions", shape=(1,), dtype=torch.int64, fill_value=0),
        BufferConfig(name="rewards", shape=(1,), dtype=torch.float32, fill_value=0.0),
        BufferConfig(name="next_observations", shape=(4,), dtype=torch.float32, fill_value=0.0),
        BufferConfig(name="dones", shape=(1,), dtype=torch.bool, fill_value=False),
    ]


@pytest.fixture
def muzero_buffer_config():
    """Real MuZero-style configuration for testing reanalyze_sequence."""
    return [
        BufferConfig(name="observations", shape=(4, 4, 3), dtype=torch.float32, fill_value=0.0),
        BufferConfig(name="actions", shape=(1,), dtype=torch.int64, fill_value=0),
        BufferConfig(name="rewards", shape=(1,), dtype=torch.float32, fill_value=0.0),
        BufferConfig(name="values", shape=(1,), dtype=torch.float32, fill_value=0.0),
        BufferConfig(name="policies", shape=(2,), dtype=torch.float32, fill_value=0.0),
        BufferConfig(name="game_ids", shape=(1,), dtype=torch.int64, fill_value=0),
        BufferConfig(name="ids", shape=(1,), dtype=torch.int64, fill_value=0),
    ]


class TestReplayBufferOperations:
    """Test insert and sample logic in the ModularReplayBuffer."""

    def test_store_single_transitions(self, tiny_buffer_config):
        """Test adding 3-4 transition dictionaries to the buffer."""
        buffer = ModularReplayBuffer(
            max_size=10,
            buffer_configs=tiny_buffer_config,
            batch_size=2,
        )

        dummy_obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        dummy_next_obs = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        transitions = [
            {
                "observations": dummy_obs,
                "actions": np.array([0], dtype=np.int64),
                "rewards": np.array([1.0], dtype=np.float32),
                "next_observations": dummy_next_obs,
                "dones": np.array([False], dtype=bool),
            },
            {
                "observations": dummy_next_obs,
                "actions": np.array([1], dtype=np.int64),
                "rewards": np.array([2.0], dtype=np.float32),
                "next_observations": np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32),
                "dones": np.array([False], dtype=bool),
            },
            {
                "observations": np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32),
                "actions": np.array([0], dtype=np.int64),
                "rewards": np.array([3.0], dtype=np.float32),
                "next_observations": np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32),
                "dones": np.array([True], dtype=bool),
            },
        ]

        for transition in transitions:
            idx = buffer.store(**transition)
            assert idx is not None

        assert buffer.size == 3

    def test_sample_returns_correct_keys_and_shapes(self, tiny_buffer_config):
        """Test that sample() returns tensors with correct keys and shapes."""
        buffer = ModularReplayBuffer(
            max_size=10,
            buffer_configs=tiny_buffer_config,
            batch_size=2,
        )

        dummy_obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        dummy_next_obs = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        for i in range(4):
            buffer.store(
                observations=dummy_obs + i,
                actions=np.array([i % 2], dtype=np.int64),
                rewards=np.array([float(i)], dtype=np.float32),
                next_observations=dummy_next_obs + i,
                dones=np.array([False], dtype=bool),
            )

        batch = buffer.sample()

        assert "observations" in batch
        assert "actions" in batch
        assert "rewards" in batch
        assert "next_observations" in batch
        assert "dones" in batch

        assert batch["observations"].shape == (2, 4)
        assert batch["actions"].shape == (2, 1)
        assert batch["rewards"].shape == (2, 1)
        assert batch["next_observations"].shape == (2, 4)
        assert batch["dones"].shape == (2, 1)

    def test_reanalyze_sequence_updates_values_and_policies(self, muzero_buffer_config):
        """Test that reanalyze_sequence correctly updates values and policies."""
        buffer = ModularReplayBuffer(
            max_size=10,
            buffer_configs=muzero_buffer_config,
            batch_size=2,
        )

        dummy_obs = np.ones((4, 4, 3), dtype=np.float32)

        for i in range(5):
            buffer.store(
                observations=dummy_obs,
                actions=np.array([i % 2], dtype=np.int64),
                rewards=np.array([float(i)], dtype=np.float32),
                values=np.array([float(i * 0.5)], dtype=np.float32),
                policies=np.array([0.5, 0.5], dtype=np.float32),
                game_ids=np.array([1], dtype=np.int64),
                ids=np.array([i + 1], dtype=np.int64),
            )

        original_values = buffer.buffers["values"][:5].clone()
        original_policies = buffer.buffers["policies"][:5].clone()

        new_values = torch.tensor([[v] for v in [10.0, 20.0, 30.0, 40.0, 50.0]], dtype=torch.float32)
        new_policies = torch.tensor(
            [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1], [0.4, 0.6]],
            dtype=torch.float32,
        )
        indices = [0, 1, 2, 3, 4]
        ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)

        buffer.reanalyze_sequence(
            indices=indices,
            new_policies=new_policies,
            new_values=new_values,
            ids=ids,
        )

        assert torch.allclose(buffer.buffers["values"][indices], new_values)
        assert torch.allclose(buffer.buffers["policies"][indices], new_policies)

        assert not torch.allclose(buffer.buffers["values"][:5], original_values)
        assert not torch.allclose(buffer.buffers["policies"][:5], original_policies)


