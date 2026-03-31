import pytest
import torch
import numpy as np

from agents.learner.batch_iterators import PPOEpochIterator
from replay_buffers.modular_buffer import ModularReplayBuffer, BufferConfig
from replay_buffers.processors import GAEProcessor, AdvantageNormalizer
from replay_buffers.writers import PPOWriter
from replay_buffers.samplers import WholeBufferSampler
from replay_buffers.concurrency import LocalBackend

pytestmark = pytest.mark.unit


def _make_ppo_buffer(max_size, num_actions, obs_dim):
    """Helper to create a filled PPO buffer for testing."""
    buffer_configs = [
        BufferConfig("observations", shape=obs_dim, dtype=torch.float32),
        BufferConfig("actions", shape=(), dtype=torch.int64),
        BufferConfig("rewards", shape=(), dtype=torch.float32),
        BufferConfig("values", shape=(), dtype=torch.float32),
        BufferConfig("policies", shape=(), dtype=torch.float32),
        BufferConfig("advantages", shape=(), dtype=torch.float32),
        BufferConfig("returns", shape=(), dtype=torch.float32),
        BufferConfig("old_log_probs", shape=(), dtype=torch.float32),
        BufferConfig("legal_moves_masks", shape=(num_actions,), dtype=torch.bool),
    ]

    gae = GAEProcessor(gamma=0.99, gae_lambda=0.95)
    normalizer = AdvantageNormalizer()

    buf = ModularReplayBuffer(
        max_size=max_size,
        batch_size=max_size,
        buffer_configs=buffer_configs,
        input_processor=gae,
        output_processor=normalizer,
        writer=PPOWriter(max_size),
        sampler=WholeBufferSampler(),
        backend=LocalBackend(),
    )

    transitions = []
    for i in range(max_size):
        transitions.append({
            "observations": np.random.randn(*obs_dim).astype(np.float32),
            "actions": np.random.randint(0, num_actions),
            "rewards": float(np.random.randn()),
            "values": float(np.random.randn() * 0.5),
            "policies": float(np.random.randn() * -1.0),
            "legal_moves_masks": np.ones(num_actions, dtype=bool),
        })

    processed = gae.process_sequence(None, transitions=transitions)
    for t in processed["transitions"]:
        buf.store(**t)

    return buf


def test_ppo_epoch_iterator_shuffling_and_minibatch():
    """
    Tier 1: PPO Mini-batching Test.
    Verifies that the PPOEpochIterator correctly shuffles data and splits it
    into the requested number of mini-batches across multiple epochs.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    max_size = 10
    num_actions = 2
    obs_dim = (4,)
    num_epochs = 2
    num_minibatches = 2
    device = torch.device("cpu")

    buffer = _make_ppo_buffer(max_size, num_actions, obs_dim)
    assert buffer.size == max_size

    iterator = PPOEpochIterator(
        replay_buffer=buffer,
        num_epochs=num_epochs,
        num_minibatches=num_minibatches,
        device=device,
    )

    yielded_batches = list(iterator)
    assert len(yielded_batches) == num_epochs * num_minibatches, (
        f"Expected {num_epochs * num_minibatches} batches, got {len(yielded_batches)}"
    )

    # Check batch sizes: each should be 5 (10 / 2)
    epoch_batches = yielded_batches[:num_minibatches]
    assert epoch_batches[0]["observations"].shape[0] == 5
    assert epoch_batches[1]["observations"].shape[0] == 5

    # Check shuffling: epoch 0 and epoch 1 should have different ordering
    epoch0_obs = torch.cat([b["observations"] for b in yielded_batches[:num_minibatches]])
    epoch1_obs = torch.cat([b["observations"] for b in yielded_batches[num_minibatches:]])
    epoch0_order = epoch0_obs[:, 0]
    epoch1_order = epoch1_obs[:, 0]
    assert not torch.equal(epoch0_order, epoch1_order), (
        "Data order was identical across epochs; shuffling might be broken."
    )


def test_ppo_epoch_iterator_uneven_split():
    """Verifies that the iterator handles cases where num_samples is not divisible by num_minibatches."""
    torch.manual_seed(42)
    np.random.seed(42)

    max_size = 11
    num_actions = 2
    obs_dim = (4,)
    num_epochs = 1
    num_minibatches = 3
    device = torch.device("cpu")

    buffer = _make_ppo_buffer(max_size, num_actions, obs_dim)

    iterator = PPOEpochIterator(
        replay_buffer=buffer,
        num_epochs=num_epochs,
        num_minibatches=num_minibatches,
        device=device,
    )

    batches = list(iterator)
    # Ceiling division: (11 + 3 - 1) // 3 = 4, so batches of 4, 4, 3
    assert len(batches) == 3
    sizes = [b["observations"].shape[0] for b in batches]
    assert sizes == [4, 4, 3], f"Expected sizes [4, 4, 3], got {sizes}"


def test_ppo_epoch_iterator_device_moving_and_norm():
    """
    Verfies that the iterator correctly moves mini-batches to the target device
    and performs advantage normalization strictly at the mini-batch level.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    max_size = 10
    num_actions = 2
    obs_dim = (4,)
    device = torch.device("cpu")

    buffer = _make_ppo_buffer(max_size, num_actions, obs_dim)

    # Sample raw batch to check that advantages are not pre-normalized
    raw_batch = buffer.sample()
    # AdvantageNormalizer normalizes at sample time, so mean should be ~0
    # But the RAW buffer values before sampling should NOT be zero-mean
    raw_advs = buffer.buffers["advantages"][:max_size]
    assert raw_advs.mean().item() != 0.0, (
        "Advantages were incorrectly normalized at the whole-batch level by the PPOBatchProcessor."
    )

    iterator = PPOEpochIterator(
        replay_buffer=buffer,
        num_epochs=1,
        num_minibatches=2,
        device=device,
    )

    for batch in iterator:
        assert batch["observations"].device.type == "cpu"
        assert batch["advantages"].device.type == "cpu"
