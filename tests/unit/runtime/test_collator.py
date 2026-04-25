import pytest
import torch
import numpy as np
from core.schema import Schema, Field, TensorSpec
from runtime.io.collator import ReplayCollator

# Set module level marker as per RULE[testing-standards.md]
pytestmark = pytest.mark.unit


def test_replay_collator_preserves_all_values():
    """
    Test 2A: Insert 32 transitions with unique actions.
    Verifies that ReplayCollator preserves all batch values and does not silently drop them.
    """
    schema = Schema([Field("action", TensorSpec(shape=(), dtype="long"))])
    collator = ReplayCollator(schema)

    # 32 transitions with unique actions
    batch = [{"action": i} for i in range(32)]
    collated = collator(batch)

    assert collated.action is not None
    assert len(collated.action) == 32
    # Check that we have all unique values 0..31
    assert torch.equal(collated.action, torch.arange(32, dtype=torch.long))


def test_replay_collator_correct_shapes_and_dtypes():
    """
    Test 2B: Verifies correct shapes and dtypes for batched data.
    obs [32, obs_dim], action [32], reward [32], done [32].
    """
    obs_dim = 8
    batch_size = 32

    schema = Schema(
        [
            Field("obs", TensorSpec(shape=(obs_dim,), dtype="float32")),
            Field("action", TensorSpec(shape=(), dtype="long")),
            Field("reward", TensorSpec(shape=(), dtype="float32")),
            Field("done", TensorSpec(shape=(), dtype="float32")),
        ]
    )
    collator = ReplayCollator(schema)

    batch = []
    for i in range(batch_size):
        batch.append(
            {
                "obs": torch.ones(obs_dim) * i,
                "action": i,
                "reward": float(i) * 0.1,
                "done": 0.0 if i < batch_size - 1 else 1.0,
                "metadata": {"step_index": i},
            }
        )

    collated = collator(batch)

    # Check shapes
    assert collated.obs.shape == (
        batch_size,
        obs_dim,
    ), f"Obs shape mismatch: {collated['obs'].shape}"
    assert collated.action.shape == (
        batch_size,
    ), f"Action shape mismatch: {collated['action'].shape}"
    assert collated.reward.shape == (
        batch_size,
    ), f"Reward shape mismatch: {collated['reward'].shape}"
    assert collated.done.shape == (
        batch_size,
    ), f"Done shape mismatch: {collated['done'].shape}"

    # Check dtypes
    assert collated.obs.dtype == torch.float32
    assert collated.action.dtype == torch.int64  # 'long' in schema maps to int64
    assert collated.reward.dtype == torch.float32
    assert collated.done.dtype == torch.float32

    # Check values
    assert collated.action[5] == 5
    assert torch.allclose(collated.reward[10], torch.tensor(1.0))

    # Check metadata (structured batch object - list of dicts in this case)
    assert isinstance(collated["metadata"], list)
    assert len(collated["metadata"]) == batch_size
    assert collated["metadata"][15]["step_index"] == 15


def test_replay_collator_empty_batch():
    """Verifies that an empty batch returns an empty dictionary."""
    schema = Schema([Field("x", TensorSpec(shape=(), dtype="float32"))])
    collator = ReplayCollator(schema)
    assert collator([]) == {}


def test_replay_collator_partial_schema():
    """Verifies that fields not in schema are still collated as lists."""
    schema = Schema([Field("obs", TensorSpec(shape=(1,), dtype="float32"))])
    collator = ReplayCollator(schema)

    batch = [
        {"obs": torch.tensor([1.0]), "extra": "a"},
        {"obs": torch.tensor([2.0]), "extra": "b"},
    ]
    collated = collator(batch)

    assert collated.obs is not None
    assert collated.metadata and "extra" in collated.metadata
    assert collated.metadata["extra"] == ["a", "b"]
